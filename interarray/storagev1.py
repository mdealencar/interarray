# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import json
import pickle
from collections.abc import Sequence
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
# already stored in database fields or are cheap to regenerate.
_misc_not = {'VertexC', 'anglesYhp', 'anglesXhp', 'anglesRank', 'angles',
             'd2rootsRank', 'd2roots', 'name', 'boundary', 'capacity',
             'runtime', 'runtime_unit', 'edges_fun', 'D', 'DetourC', 'fnT',
             'landscape_angle', 'Root', 'creation_options', 'G_nodeset',
             'non_A_gates', 'funfile', 'funhash', 'funname', 'diagonals',
             'planar', 'has_loads', 'R', 'Subtree', 'handle', 'non_A_edges',
             'max_load', 'fun_fingerprint', 'hull', 'solver_log',
             'loading_length_mismatch', 'gnT', 'C', 'border'}


def base_graph_from_nodeset(nodeset: object) -> nx.Graph:
    '''Create the networkx Graph (nodes only) for a given nodeset.'''
    T = nodeset.T
    R = nodeset.R
    G = nx.Graph(name=nodeset.name,
                 T=T,
                 R=R,
                 VertexC=pickle.loads(nodeset.VertexC),
                 boundary=pickle.loads(nodeset.boundary),
                 landscape_angle=nodeset.landscape_angle,
                 )
    G.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                      for n in range(T)))
    G.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                      for r in range(-R, 0)))
    return G


def graph_from_edgeset(edgeset: object) -> nx.Graph:
    nodeset = edgeset.nodes
    G = base_graph_from_nodeset(nodeset)
    G.graph.update(handle=edgeset.handle,
                   capacity=edgeset.capacity,
                   funhash=edgeset.method.funhash,
                   funfile=edgeset.method.funfile,
                   funname=edgeset.method.funname,
                   runtime=edgeset.runtime,
                   creation_options=edgeset.method.options,
                   **edgeset.misc)

    add_edges_to(G, edges=edgeset.edges, clone2prime=edgeset.clone2prime)
    calc_length = G.size(weight='length')
    #  assert abs(calc_length/edgeset.length - 1) < 1e-5, (
    #      f"recreated graph's total length ({calc_length:.0f}) != "
    #      f"stored total length ({edgeset.length:.0f})")
    if abs(calc_length/edgeset.length - 1) > 1e-5:
        G.graph['loading_length_mismatch'] = calc_length - edgeset.length
    return G


def packnodes(G: nx.Graph) -> PackType:
    R = G.graph['R']
    D = G.graph.get('D', 0)
    T = G.number_of_nodes() - D - R
    digest, pickled_coordinates = site_fingerprint(G.graph['VertexC'],
                                                   G.graph['boundary'])
    if G.name[0] == '!':
        name = G.name + base64.b64encode(digest).decode('ascii')
    else:
        name = G.name
    pack = dict(
        digest=digest,
        name=name,
        T=T,
        R=R,
        landscape_angle=G.graph.get('landscape_angle', 0.),
        **pickled_coordinates,
    )
    return pack


def packmethod(ffprint: dict, options: Mapping | None = None) -> PackType:
    options = options or {}
    if 'capacity' in options:
        del options['capacity']
    #  funhash = sha256(fun.__code__.co_code).digest()
    funhash = ffprint['funhash']
    optionsstr = json.dumps(options)
    digest = sha256(funhash + optionsstr.encode()).digest()
    pack = dict(
        digest=digest,
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


def method_from_graph(G: nx.Graph, db: orm.Database) -> bytes:
    '''Returns primary key of the entry.'''
    pack = packmethod(G.graph['fun_fingerprint'], G.graph['creation_options'])
    return add_if_absent(db.Method, pack)


def nodeset_from_graph(G: nx.Graph, db: orm.Database) -> bytes:
    '''Returns primary key of the entry.'''
    pack = packnodes(G)
    return add_if_absent(db.NodeSet, pack)


def terse_graph_from_G(G: nx.Graph) -> dict:
    '''
    Returns a dict with:
        edges: where ⟨i, edge[i]⟩ is a directed edge
        clone2prime: mapping the above-T clones to below-T nodes
    '''
    R = G.graph['R']
    V = G.number_of_nodes() - R
    edges = np.empty((V,), dtype=int)
    if not G.graph.get('has_loads'):
        calcload(G)
    for u, v, reverse in G.edges(data='reverse'):
        u, v = (u, v) if u < v else (v, u)
        i, target = (u, v) if reverse else (v, u)
        edges[i] = target
    terse_graph = dict(edges=edges)
    D = G.graph.get('D', 0)
    if D > 0:
        T = V - D
        terse_graph['clone2prime'] = G.graph['fnT'][T: T + D]
    return terse_graph


def add_edges_to(G: nx.Graph, edges: np.ndarray,
                 clone2prime: np.ndarray | None = None) -> None:
    '''
    Changes G in place if it has no edges, else copies G nodes and graph
    attributes.
    '''
    if G.number_of_edges() > 0:
        G = nx.create_empty_copy(G, with_data=True)
    VertexC = G.graph['VertexC']
    R = G.graph['R']
    T = G.graph['T']
    if clone2prime:
        D = len(clone2prime)
        G.graph['D'] = D
        detournodes = range(T, T + D)
        G.add_nodes_from(((s, {'kind': 'detour'})
                          for s in detournodes))
        fnT = np.arange(T + D + R)
        fnT[T: T + D] = clone2prime
        fnT[-R:] = range(-R, 0)
        G.graph['fnT'] = fnT
        AllnodesC = np.vstack((VertexC[:T],
                               VertexC[clone2prime],
                               VertexC[-R:]))
    else:
        D = 0
        AllnodesC = VertexC
    Length = np.hypot(*(AllnodesC[:-R] - AllnodesC[edges]).T)
    G.add_weighted_edges_from(zip(range(T + D), edges, Length),
                              weight='length')
    if D > 0:
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


def packedges(G: nx.Graph) -> dict[str, Any]:
    R = G.graph['R']
    T = G.graph['VertexC'].shape[0] - R
    terse_graph = terse_graph_from_G(G)
    misc = {key: G.graph[key]
            for key in G.graph.keys() - _misc_not}
    #  print('Storing in `misc`:', *misc.keys())
    for k, v in misc.items():
        misc[k] = oddtypes_to_serializable(v)
    edgepack = dict(
        handle=G.graph.get('handle',
                           G.graph['name'].strip().replace(' ', '_')),
        capacity=G.graph['capacity'],
        length=G.size(weight='length'),
        runtime=G.graph['runtime'],
        gates=[len(G[root]) for root in range(-R, 0)],
        T=T,
        R=R,
        misc=misc,
        D=G.graph.get('D', 0),
        **terse_graph,
    )
    return edgepack


def edgeset_from_graph(G: nx.Graph, db: orm.Database) -> int:
    '''Add a new EdgeSet entry in the database, using the data in `G`.
    If the NodeSet or Method are not yet in the database, they will be added.

    Return value: primary key of the newly created EdgeSet record
    '''
    edgepack = packedges(G)
    nodesetID = nodeset_from_graph(G, db)
    methodID = method_from_graph(G, db),
    machineID = get_machine_pk(db)
    with orm.db_session:
        edgepack.update(
            nodes=db.NodeSet[nodesetID],
            method=db.Method[methodID],
            machine=db.Machine[machineID],
        )
        return db.EdgeSet(**edgepack).get_pk()


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
