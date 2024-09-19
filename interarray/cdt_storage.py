# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import json
import pickle
from collections.abc import Sequence
from itertools import pairwise
from hashlib import sha256
import base64
from socket import getfqdn, gethostname
from typing import Any, Mapping

import networkx as nx
import numpy as np
from pony import orm

from .interarraylib import calcload, site_fingerprint
from .utils import NodeTagger

F = NodeTagger()

PackType = Mapping[str, Any]

# Set of not-to-store keys commonly found in G routesets (they are either
# already stored in database fields or are cheap to regenerate or too big.
_misc_not = {'VertexC', 'anglesYhp', 'anglesXhp', 'anglesRank', 'angles',
             'd2rootsRank', 'd2roots', 'name', 'boundary', 'capacity',
             'runtime', 'runtime_unit', 'edges_fun', 'D', 'DetourC', 'fnT',
             'landscape_angle', 'Root', 'creation_options', 'G_nodeset',
             'non_A_gates', 'funfile', 'funhash', 'funname', 'diagonals',
             'planar', 'has_loads', 'M', 'Subtree', 'handle', 'non_A_edges',
             'max_load', 'fun_fingerprint', 'overfed', 'hull', 'solver_log',
             'length_mismatch_on_db_read', 'gnT', 'C', 'border', 'exclusions',
             'diagonals_used', 'crossings_map', 'tentative', 'creator',
             'is_normalized'}


def S_from_nodeset(nodeset: object) -> nx.Graph:
    '''Create the networkx Graph (nodes only) for a given nodeset.'''
    N = nodeset.N
    M = nodeset.M
    B, *exclusion_groups = nodeset.constraint_groups
    border = nodeset.constraint_vertices[:B]
    S = nx.Graph(
         M=M, N=N, B=B,
         name=nodeset.name,
         border=border,
         VertexC=pickle.loads(nodeset.VertexC),
         boundary=pickle.loads(nodeset.boundary),
         landscape_angle=nodeset.landscape_angle,
    )
    if exclusion_groups:
        S.graph.update(
            exclusions=[nodeset.constraint_vertices[a:b] for a, b in
                        pairwise([B] + exclusion_groups + [None])])
    S.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                      for n in range(N)))
    S.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                      for r in range(-M, 0)))
    return S


def G_from_routeset(routeset: object) -> nx.Graph:
    nodeset = routeset.nodes
    M, N, B = nodeset.M, nodeset.N, nodeset.B
    G = S_from_nodeset(nodeset)
    G.graph.update(
        M=M, N=N, B=B,
        handle=routeset.handle,
        capacity=routeset.capacity,
        funhash=routeset.method.funhash,
        funfile=routeset.method.funfile,
        funname=routeset.method.funname,
        runtime=routeset.runtime,
        creation_options=routeset.method.options,
        **routeset.misc)

    if routeset.stuntC:
        stuntC = pickle.loads(routeset.stuntC)
        G.graph['B'] += len(stuntC)
        VertexC = G.graph['VertexC']
        G.graph['VertexC'] = np.vstack((VertexC[:-M], stuntC,
                                        VertexC[-M:]))
    add_edges_to(G, edges=routeset.edges, clone2prime=routeset.clone2prime)
    G.graph['overfed'] = [len(G[root])/np.ceil(N/routeset.capacity)*M
                          for root in range(-M, 0)]
    calc_length = G.size(weight='length')
    #  assert abs(calc_length/routeset.length - 1) < 1e-5, (
    #      f"recreated graph's total length ({calc_length:.0f}) != "
    #      f"stored total length ({routeset.length:.0f})")
    if abs(calc_length/routeset.length - 1) > 1e-5:
        G.graph['length_mismatch_on_db_read'] = calc_length - routeset.length
    if routeset.tentative:
        for r, n in pairwise(routeset.tentative):
            G[r][n]['kind'] = 'tentative'
    return G


def packnodes(G: nx.Graph) -> PackType:
    M, N, B = (G.graph[k] for k in 'MNB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC']
    # border_stunts, stuntC
    border_stunts = G.graph.get('border_stunts')
    if border_stunts:
        VertexC = np.vstack((VertexC[:N + B - len(border_stunts)],
                             VertexC[-M:]))
    VertexCpkl = pickle.dumps(VertexC)
    digest = sha256(VertexCpkl).digest(),

    if G.name[0] == '!':
        name = G.name + base64.b64encode(digest).decode('ascii')
    else:
        name = G.name
    constraint_vertices = list(chain((G.graph.get('border', ()),),
                                     G.graph.get('exclusions', ())))
    pack = dict(
        N=N, M=M,
        name=name,
        VertexC=VertexCpkl,
        constraint_groups=[len(p) for p in constraint_vertices],
        constraint_vertices=sum(constraint_vertices, []),
        landscape_angle=G.graph.get('landscape_angle', 0.),
        digest=digest,
    )
    return pack


def packmethod(method_options: dict) -> PackType:
    options = method_options.copy()
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


def terse_graph_from_G(G: nx.Graph) -> PackType:
    '''Convert `G`'s edges to a format suitable for storing in the database.

    Although graph `G` in undirected, the edge attribute `'reverse'` and its
    nodes' numbers encode the direction of power flow. The terse
    representation uses that and the fact that `G` is a tree.

    Returns:
        dict with keys:
            edges: where ⟨i, edges[i]⟩ is a directed edge of `G`
            clone2prime: mapping the above-N clones to below-N nodes
    '''
    M, N, B = (G.graph[k] for k in 'MNB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    edges = np.empty((N + C + D,), dtype=int)
    if not G.graph.get('has_loads'):
        calcload(G)
    for u, v, reverse in G.edges(data='reverse'):
        u, v = (u, v) if u < v else (v, u)
        i, target = (u, v) if reverse else (v, u)
        if i < N:
            edges[i] = target
        else:
            edges[i - B] = target
    terse_graph = dict(edges=edges)
    if C > 0 or D > 0:
        terse_graph['clone2prime'] = G.graph['fnT'][N + B: -M]
    return terse_graph


def add_edges_to(G: nx.Graph, edges: np.ndarray,
                 clone2prime: np.ndarray | None = None) -> None:
    '''
    Changes G in place if it has no edges, else copies G nodes and graph
    attributes.
    '''
    if G.number_of_edges() > 0:
        G = nx.create_empty_copy(G, with_data=True)
    M, N, B = (G.graph[k] for k in 'MNB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC']
    source = np.arange(len(edges))
    target = edges.copy()
    if clone2prime:
        source[N:] += B
        contournodes = range(N + B, N + B + C)
        detournodes = range(N + B + C, N + B + C + D)
        G.add_nodes_from(((s, {'kind': 'contour'})
                          for s in detournodes))
        G.add_nodes_from(((s, {'kind': 'detour'})
                          for s in detournodes))
        fnT = np.arange(M + N + B + C + D)
        fnT[N + B: N + B + C + D] = clone2prime
        fnT[-M:] = range(-M, 0)
        G.graph['fnT'] = fnT
    Length = np.hypot(*(AllnodesC[target] - AllnodesC[source]).T)
    G.add_weighted_edges_from(zip(source, target, Length),
                              weight='length')
    if clone2prime:
        for _, _, edgeD in G.edges(contournodes, data=True):
            edgeD['kind'] = 'contour'
        for _, _, edgeD in G.edges(detournodes, data=True):
            edgeD['kind'] = 'detour'
    calcload(G)
    return G


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
    else:
        return obj


def pack_G(G: nx.Graph) -> dict[str, Any]:
    M, N, B = (G.graph[k] for k in 'MNB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    terse_graph = terse_graph_from_G(G)
    misc = {key: G.graph[key]
            for key in G.graph.keys() - _misc_not}
    #  print('Storing in `misc`:', *misc.keys())
    for k, v in misc.items():
        misc[k] = oddtypes_to_serializable(v)
    length = G.size(weight='length')
    packed_G = dict(
        M=M, N=N, B=B, C=C, D=D,
        handle=G.graph.get('handle',
                           G.graph['name'].strip().replace(' ', '_')),
        capacity=G.graph['capacity'],
        length=length,
        creator=G.graph['creator'],
        is_normalized=G.graph['is_normalized'],
        runtime=G.graph['runtime'],
        num_gates=[len(G[root]) for root in range(-M, 0)],
        misc=misc,
        **terse_graph,
    )
    # border_stunts, stuntC
    border_stunts = G.graph.get('border_stunts')
    if border_stunts:
        stuntC = VertexC[N + B - len(border_stunts): N + B].copy()
        packed_G['stuntC'] = pickle.dumps(stuntC)
    objective = G.graph.get('objective')
    if D > 0 and objective is not None:
        packed_G['detextra'] = length/objective - 1
    diagonals_used = G.graph.get('diagonals_used')
    if diagonals_used is not None:
        packed_G['diagonals_used'] = diagonals_used
    tentative = G.graph.get('tentative')
    if tentative is not None:
        # edges are concatenated in a single array of nodes
        packed_G['tentative'] = sum(tentative, ())
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
        return db.RouteSet(**packed_G).get_pk()


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
    es = db.EdgeSet.get(lambda e:
                        e.nodes.name == farmname and
                        e.method is method and
                        e.capacity == c)
    Gdb = graph_from_edgeset(es)
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
            graph_from_edgeset(
                db.EdgeSet.get(lambda e:
                               e.nodes.name == farm.name and
                               e.method is m and
                               e.capacity == c))
            for m in methods)
        for G in Gtuple:
            calcload(G)
        Gs.append(Gtuple)
    return Gs
