# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import json
import pickle
from collections.abc import Sequence
from hashlib import sha256
import base64
from socket import getfqdn, gethostname
from typing import Any, List, Mapping, Optional, Tuple, Union

import networkx as nx
import numpy as np
from pony import orm

from .interarraylib import calcload, site_fingerprint
from .utils import NodeTagger

# Coordinates use arrays of floats.
# Somehow, nodesets with the same coordinates were getting different digests,
# when the code ran on different computers.
# Rouding to a fixed (small) number of decimal place to fix it.
COORDINATES_DECIMAL_PLACES = 2

F = NodeTagger()

PackType = Mapping[str, Any]


def base_graph_from_nodeset(nodeset: object) -> nx.Graph:
    '''Create the networkx Graph (nodes only) for a given nodeset.'''
    N = nodeset.N
    M = nodeset.M
    G = nx.Graph(name=nodeset.name,
                 M=M,
                 VertexC=pickle.loads(nodeset.VertexC),
                 boundary=pickle.loads(nodeset.boundary),
                 landscape_angle=nodeset.landscape_angle,
                 )
    G.add_nodes_from(((n, {'label': F[n], 'type': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'type': 'oss'})
                      for r in range(-M, 0)))
    return G


def graph_from_edgeset(edgeset: object) -> nx.Graph:
    nodeset = edgeset.nodes
    VertexC = pickle.loads(nodeset.VertexC)
    N = nodeset.N
    M = nodeset.M
    G = nx.Graph(name=nodeset.name,
                 handle=edgeset.handle,
                 M=M,
                 VertexC=VertexC,
                 capacity=edgeset.capacity,
                 boundary=pickle.loads(nodeset.boundary),
                 landscape_angle=nodeset.landscape_angle,
                 funhash=edgeset.method.funhash,
                 funfile=edgeset.method.funfile,
                 funname=edgeset.method.funname,
                 creation_options=edgeset.method.options,
                 **edgeset.misc)

    G.add_nodes_from(((n, {'label': F[n], 'type': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'type': 'oss'})
                      for r in range(-M, 0)))

    D = edgeset.D or 0
    if D > 0:
        G.graph['D'] = D
        detournodes = range(N, N + D)
        G.add_nodes_from(((s, {'type': 'detour'})
                          for s in detournodes))
        clone2prime = edgeset.clone2prime
        assert len(clone2prime) == D, \
            'len(EdgeSet.clone2prime) != EdgeSet.D'
        fnT = np.arange(N + D + M)
        fnT[N: N + D] = clone2prime
        fnT[-M:] = range(-M, 0)
        G.graph['fnT'] = fnT
        AllnodesC = np.vstack((VertexC[:N],
                               VertexC[clone2prime],
                               VertexC[-M:]))
    else:
        AllnodesC = VertexC

    Length = np.hypot(*(AllnodesC[:-M] - AllnodesC[edgeset.edges]).T)
    G.add_weighted_edges_from(zip(range(N + D), edgeset.edges, Length),
                              weight='length')
    if D > 0:
        for _, _, edgeD in G.edges(detournodes, data=True):
            edgeD['type'] = 'detour'
    G.graph['overfed'] = [len(G[root])/np.ceil(N/edgeset.capacity)*M
                          for root in range(-M, 0)]
    calc_length = G.size(weight='length')
    assert abs(calc_length - edgeset.length) < 1, (
        f"recreated graph's total length ({calc_length:.0f}) != "
        f"stored total length ({edgeset.length:.0f})")

    # make_graph_metrics(G)
    calcload(G)
    return G


def packnodes(G: nx.Graph) -> PackType:
    M = G.graph['M']
    D = G.graph.get('D', 0)
    N = G.number_of_nodes() - D - M
    digest, pickled_coordinates = site_fingerprint(G.graph['VertexC'],
                                                   G.graph['boundary'])
    if G.name[0] == '!':
        name = G.name + base64.b64encode(digest).decode('ascii')
    else:
        name = G.name
    pack = dict(
        digest=digest,
        name=name,
        N=N,
        M=M,
        landscape_angle=G.graph.get('landscape_angle', 0.),
        **pickled_coordinates,
    )
    return pack


def packmethod(ffprint: dict, options: Optional[Mapping] = None) -> PackType:
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


def edgeset_from_graph(G: nx.Graph, db: orm.Database) -> int:
    '''Add a new EdgeSet entry in the database, using the data in `G`.
    If the NodeSet or Method are not yet in the database, they will be added.

    Return value: primary key of the newly created EdgeSet record
    '''
    misc_not = {'VertexC', 'anglesYhp', 'anglesXhp', 'anglesRank', 'angles',
                'd2rootsRank', 'd2roots', 'name', 'boundary', 'capacity',
                'runtime', 'runtime_unit', 'edges_fun', 'D', 'DetourC', 'fnT',
                'crossings', 'landscape_angle', 'Root', 'creation_options', 
                'Subtree', 'funfile', 'funhash', 'funname', 'diagonals',
                'planar', 'has_loads', 'M', 'gates_not_in_A', 'overfed',
                'gnT', 'fun_fingerprint', 'handle', 'hull'}
    nodesetID = nodeset_from_graph(G, db)
    methodID = method_from_graph(G, db),
    machineID = get_machine_pk(db)
    M = G.graph['M']
    N = G.graph['VertexC'].shape[0] - M
    V = G.number_of_nodes() - M
    edges = np.empty((V,), dtype=int)
    if not G.graph.get('has_loads'):
        calcload(G)
    for u, v, reverse in G.edges(data='reverse'):
        u, v = (u, v) if u < v else (v, u)
        i, target = (u, v) if reverse else (v, u)
        edges[i] = target
    misc = {key: G.graph[key]
            for key in G.graph.keys() - misc_not}
    print('Storing in `misc`:', *misc.keys())
    for k, v in misc.items():
        if isinstance(v, np.ndarray):
            misc[k] = v.tolist()
        elif isinstance(v, np.int64):
            misc[k] = int(v)
    edgepack = dict(
            handle=G.graph.get('handle',
                               G.graph['name'].strip().replace(' ', '_')),
            capacity=G.graph['capacity'],
            length=G.size(weight='length'),
            runtime=G.graph['runtime'],
            gates=[len(G[root]) for root in range(-M, 0)],
            N=N,
            M=M,
            misc=misc,
            edges=edges,
    )
    D = G.graph.get('D')
    if D is not None and D > 0:
        N = V - D
        edgepack['D'] = D
        edgepack['clone2prime'] = G.graph['fnT'][N: N + D]
    else:
        edgepack['D'] = 0
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


def Gs_from_attrs(farm: object, methods: Union[object, Sequence[object]],
                  capacities: Union[int, Sequence[int]],
                  db: orm.Database) -> List[Tuple[nx.Graph]]:
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
