# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import io
import json
from collections.abc import Sequence
from functools import partial
from itertools import pairwise, chain
from hashlib import sha256
import base64
from socket import getfqdn, gethostname
from typing import Any, Mapping

import networkx as nx
import numpy as np
from pony import orm

from .interarraylib import calcload
from .utils import NodeTagger

F = NodeTagger()

PackType = Mapping[str, Any]

# Set of not-to-store keys commonly found in G routesets (they are either
# already stored in database fields or are cheap to regenerate or too big.
_misc_not = {'VertexC', 'anglesYhp', 'anglesXhp', 'anglesRank', 'angles',
             'd2rootsRank', 'd2roots', 'name', 'boundary', 'capacity', 'B',
             'runtime', 'runtime_unit', 'edges_fun', 'D', 'DetourC', 'fnT',
             'landscape_angle', 'Root', 'creation_options', 'G_nodeset', 'T',
             'non_A_gates', 'funfile', 'funhash', 'funname', 'diagonals',
             'planar', 'has_loads', 'R', 'Subtree', 'handle', 'non_A_edges',
             'max_load', 'fun_fingerprint', 'hull', 'solver_log',
             'length_mismatch_on_db_read', 'gnT', 'C', 'border', 'obstacles',
             'num_diagonals', 'crossings_map', 'tentative', 'method_options',
             'is_normalized', 'norm_scale', 'norm_offset', 'detextra', 'rogue',
             'clone2prime', 'valid', 'path_in_P', 'shortened_contours',
             'nonAedges', 'method', 'num_stunts', 'crossings', 'creator'}


def L_from_nodeset(nodeset: object) -> nx.Graph:
    '''Create the networkx Graph (nodes only) for a given nodeset.'''
    T = nodeset.T
    R = nodeset.R
    # assert B == sum(n >= T for n in nodeset.constraint_vertices)
    B = nodeset.B
    border = np.array(nodeset.constraint_vertices[:nodeset.constraint_groups[0]])
    name=nodeset.name
    L = nx.Graph(
         R=R, T=T, B=B,
         name=name,
         handle=((name if name[0] != '!' else name[1:name.index('!', 1)])
                 .strip().lower().replace(' ', '_')),
         border=border,
         VertexC=np.lib.format.read_array(io.BytesIO(nodeset.VertexC)),
         landscape_angle=nodeset.landscape_angle,
    )
    if len(nodeset.constraint_groups) > 1:
        obstacle_idx = np.cumsum(np.array(nodeset.constraint_groups))
        L.graph.update(
            obstacles=[np.array(nodeset.constraint_vertices[a:b])
                       for a, b in pairwise(obstacle_idx)])
    L.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                      for n in range(T)))
    L.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                      for r in range(-R, 0)))
    return L


def G_from_routeset(routeset: object) -> nx.Graph:
    nodeset = routeset.nodes
    R = nodeset.R
    G = L_from_nodeset(nodeset)
    G.graph.update(
        C=routeset.C, D=routeset.D,
        handle=routeset.handle,
        capacity=routeset.capacity,
        creator=routeset.creator,
        method=dict(
            solver_name=routeset.method.solver_name,
            timestamp=routeset.method.timestamp,
            funname=routeset.method.funname,
            funfile=routeset.method.funfile,
            funhash=routeset.method.funhash,
        ),
        runtime=routeset.runtime,
        method_options=routeset.method.options,
        **routeset.misc)

    if routeset.detextra is not None:
        G.graph['detextra'] = routeset.detextra

    if routeset.stuntC:
        stuntC=np.lib.format.read_array(io.BytesIO(routeset.stuntC))
        num_stunts = len(stuntC)
        G.graph['num_stunts'] = num_stunts
        G.graph['B'] += num_stunts
        VertexC = G.graph['VertexC']
        G.graph['VertexC'] = np.vstack((VertexC[:-R], stuntC,
                                        VertexC[-R:]))
    untersify_to_G(G, terse=routeset.edges, clone2prime=routeset.clone2prime)
    calc_length = G.size(weight='length')
    #  assert abs(calc_length/routeset.length - 1) < 1e-5, (
    #      f"recreated graph's total length ({calc_length:.0f}) != "
    #      f"stored total length ({routeset.length:.0f})")
    if abs(calc_length/routeset.length - 1) > 1e-5:
        G.graph['length_mismatch_on_db_read'] = calc_length - routeset.length
    if routeset.rogue:
        for u, v in zip(routeset.rogue[::2], routeset.rogue[1::2]):
            G[u][v]['kind'] = 'rogue'
    if routeset.tentative:
        for r, n in zip(routeset.tentative[::2], routeset.tentative[1::2]):
            G[r][n]['kind'] = 'tentative'
    return G


def packnodes(G: nx.Graph) -> PackType:
    R, T, B = (G.graph[k] for k in 'RTB')
    VertexC = G.graph['VertexC']
    num_stunts = G.graph.get('num_stunts')
    if num_stunts:
        B -= num_stunts
        VertexC = np.vstack((VertexC[:T + B],
                             VertexC[-R:]))
    VertexC_npy_io = io.BytesIO()
    np.lib.format.write_array(VertexC_npy_io, VertexC, version=(3, 0))
    VertexC_npy = VertexC_npy_io.getvalue()
    digest = sha256(VertexC_npy).digest()

    if G.name[0] == '!':
        name = G.name + base64.b64encode(digest).decode('ascii')
    else:
        name = G.name
    constraint_vertices = list(chain((G.graph.get('border', ()),),
                                     G.graph.get('obstacles', ())))
    pack = dict(
        T=T, R=R, B=B,
        name=name,
        VertexC=VertexC_npy,
        constraint_groups=[p.shape[0] for p in constraint_vertices],
        constraint_vertices=np.concatenate(constraint_vertices,
                                           dtype=int, casting='unsafe'),
        landscape_angle=G.graph.get('landscape_angle', 0.),
        digest=digest,
    )
    return pack


def packmethod(method_options: dict) -> PackType:
    options = method_options.copy()
    gates_limit = options.get('gates_limit')
    if isinstance(gates_limit, int):
        options['gates_limit'] = 'given'
    ffprint = options.pop('fun_fingerprint')
    solver_name = options.pop('solver_name')
    optionsstr = json.dumps(options)
    digest = sha256(ffprint['funhash'] + optionsstr.encode()).digest()
    pack = dict(
        digest=digest,
        solver_name=solver_name,
        options=options,
        **ffprint,
    )
    return pack


def add_if_absent(entity: object, pack: PackType) -> bytes:
    digest = pack['digest']
    with orm.db_session:
        if not entity.exists(digest=digest):
            entity(**pack)
    return digest


def method_from_G(G: nx.Graph, db: orm.Database) -> bytes:
    '''
    Returns:
        Primary key of the entry.
    '''
    pack = packmethod(G.graph['method_options'])
    return add_if_absent(db.Method, pack)


def nodeset_from_G(G: nx.Graph, db: orm.Database) -> bytes:
    '''Returns primary key of the entry.'''
    pack = packnodes(G)
    return add_if_absent(db.NodeSet, pack)


def terse_pack_from_G(G: nx.Graph) -> PackType:
    '''Convert `G`'s edges to a format suitable for storing in the database.

    Although graph `G` in undirected, the edge attribute `'reverse'` and its
    nodes' numbers encode the direction of power flow. The terse
    representation uses that and the fact that `G` is a tree.

    Returns:
        dict with keys:
            edges: where ⟨i, edges[i]⟩ is a directed edge of `G`
            clone2prime: mapping the above-T clones to below-T nodes
    '''
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    terse = np.empty((T + C + D,), dtype=int)
    if not G.graph.get('has_loads'):
        calcload(G)
    for u, v, reverse in G.edges(data='reverse'):
        if reverse is None:
            raise ValueError('reverse must not be None')
        u, v = (u, v) if u < v else (v, u)
        i, target = (u, v) if reverse else (v, u)
        if i < T:
            terse[i] = target
        else:
            terse[i - B] = target
    terse_pack = dict(edges=terse)
    if C > 0 or D > 0:
        terse_pack['clone2prime'] = G.graph['fnT'][T + B: -R]
    return terse_pack


def untersify_to_G(G: nx.Graph, terse: np.ndarray,
                   clone2prime: list) -> None:
    '''
    Changes G in place!
    '''
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC']
    source = np.arange(len(terse))
    if clone2prime:
        source[T:] += B
        contournodes = range(T + B, T + B + C)
        detournodes = range(T + B + C, T + B + C + D)
        G.add_nodes_from(contournodes, kind='contour')
        G.add_nodes_from(detournodes, kind='detour')
        fnT = np.arange(R + T + B + C + D)
        fnT[T + B: T + B + C + D] = clone2prime
        fnT[-R:] = range(-R, 0)
        G.graph['fnT'] = fnT
        Length = np.hypot(*(VertexC[fnT[terse]] - VertexC[fnT[source]]).T)
    else:
        Length = np.hypot(*(VertexC[terse] - VertexC[source]).T)
    G.add_weighted_edges_from(
        zip(source.tolist(), terse, Length.tolist()), weight='length')
    if clone2prime:
        for _, _, edgeD in G.edges(contournodes, data=True):
            edgeD['kind'] = 'contour'
        for _, _, edgeD in G.edges(detournodes, data=True):
            edgeD['kind'] = 'detour'
    calcload(G)


def oddtypes_to_serializable(obj):
    if isinstance(obj, orm.ormtypes.TrackedList):
        return list(oddtypes_to_serializable(item) for item in obj)
    if isinstance(obj, orm.ormtypes.TrackedDict):
        return {k: oddtypes_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(oddtypes_to_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    else:
        return obj


def pack_G(G: nx.Graph) -> dict[str, Any]:
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    terse_pack = terse_pack_from_G(G)
    misc = {key: G.graph[key]
            for key in G.graph.keys() - _misc_not}
    #  print('Storing in `misc`:', *misc.keys())
    for k, v in misc.items():
        misc[k] = oddtypes_to_serializable(v)
    length = G.size(weight='length')
    packed_G = dict(
        R=R, T=T, C=C, D=D,
        handle=G.graph.get('handle',
                           G.graph['name'].strip().replace(' ', '_')),
        capacity=G.graph['capacity'],
        length=length,
        creator=G.graph['creator'],
        is_normalized=G.graph.get('is_normalized', False),
        runtime=G.graph['runtime'],
        num_gates=[len(G[root]) for root in range(-R, 0)],
        misc=misc,
        **terse_pack,
    )
    # Optional fields
    num_stunts = G.graph.get('num_stunts')
    if num_stunts:
        VertexC = G.graph['VertexC']
        stuntC = VertexC[T + B - num_stunts: T + B].copy()
        stuntC_npy_io = io.BytesIO()
        np.lib.format.write_array(stuntC_npy_io, stuntC, version=(3, 0))
        packed_G['stuntC'] = stuntC_npy_io.getvalue()
    concatenate_tuples = partial(sum, start=())
    pack_if_given = (  # key, function to prepare data
        ('detextra', None),
        ('num_diagonals', None),
        ('valid', None),
        ('tentative', concatenate_tuples),
        ('rogue', concatenate_tuples),
    )
    packed_G.update({k: (fun(G.graph[k]) if fun else G.graph[k])
                     for k, fun in pack_if_given if k in G.graph})
    return packed_G


def store_G(G: nx.Graph, db: orm.Database) -> int:
    '''Store `G`'s data to a new `RouteSet` record in the database `db`.

    If the NodeSet or Method are not yet in the database, they will be added.

    Args:
        G: Graph with the routeset.
        db: Database instance.

    Returns:
        Primary key of the newly created RouteSet record.
    '''
    packed_G = pack_G(G)
    nodesetID = nodeset_from_G(G, db)
    methodID = method_from_G(G, db),
    machineID = get_machine_pk(db)
    with orm.db_session:
        packed_G.update(
            nodes=db.NodeSet[nodesetID],
            method=db.Method[methodID],
            machine=db.Machine[machineID],
        )
        rs = db.RouteSet(**packed_G)
        db.flush()
        id = rs.id
    return id


def get_machine_pk(db: orm.Database) -> int:
    fqdn = getfqdn()
    hostname = gethostname()
    if fqdn == 'localhost':
        machine = hostname
    else:
        if hostname.startswith('n-'):
            machine = fqdn[len(hostname):]
        else:
            machine = fqdn
    with orm.db_session:
        if db.Machine.exists(name=machine):
            return db.Machine.get(name=machine).get_pk()
        else:
            return db.Machine(name=machine).get_pk()


def G_by_method(G: nx.Graph, method: object, db: orm.Database) -> nx.Graph:
    '''Fetch from the database a layout for `G` by `method`.
    `G` must be a layout solution with the necessary info in the G.graph dict.
    `method` is a Method.
    '''
    farmname = G.name
    c = G.graph['capacity']
    rs = db.RouteSet.get(lambda rs:
                         rs.nodes.name == farmname and
                         rs.method is method and
                         rs.capacity == c)
    Gdb = G_from_routeset(rs)
    calcload(Gdb)
    return Gdb


def Gs_from_attrs(farm: object, methods: object | Sequence[object],
                  capacities: int | Sequence[int],
                  db: orm.Database) -> list[tuple[nx.Graph]]:
    '''
    Fetch from the database a list (one per capacity) of tuples (one per
    method) of layouts.
    `farm` must have the desired NodeSet name in the `name` attribute.
    `methods` is a (sequence of) Method instance(s).
    `capacities` is a (sequence of) int(s).
    '''
    Gs = []
    if not isinstance(methods, Sequence):
        methods = (methods,)
    if not isinstance(capacities, Sequence):
        capacities = (capacities,)
    for c in capacities:
        Gtuple = tuple(
            G_from_routeset(
                db.RouteSet.get(lambda rs:
                                rs.nodes.name == farm.name and
                                rs.method is m and
                                rs.capacity == c))
            for m in methods)
        for G in Gtuple:
            calcload(G)
        Gs.append(Gtuple)
    return Gs
