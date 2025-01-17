# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import json
import pickle
from collections.abc import Sequence
from hashlib import sha256
from socket import getfqdn, gethostname

import networkx as nx
import numpy as np
from pony.orm import db_session

from .interarraylib import calcload
from .utils import NodeTagger

# Coordinates use arrays of floats.
# Somehow, nodesets with the same coordinates were getting different digests,
# when the code ran on different computers.
# Rouding to a fixed (small) number of decimal place to fix it.
COORDINATES_DECIMAL_PLACES = 2

F = NodeTagger()


def graph_from_edgeset(edgeset):
    nodeset = edgeset.nodes
    VertexC = pickle.loads(nodeset.VertexC)
    T = nodeset.T
    R = nodeset.R
    pickled_misc = edgeset.misc
    if pickled_misc is None:
        creator = edgeset.method.funname
        iterations = 1
    else:
        misc = pickle.loads(pickled_misc)
        creator = misc['edges_created_by']
        iterations = misc.get('iterations', 1)
    G = nx.Graph(name=nodeset.name,
                 R=R, T=T, handle=nodeset.handle,
                 VertexC=VertexC,
                 capacity=edgeset.capacity,
                 boundary=pickle.loads(nodeset.boundary),
                 landscape_angle=nodeset.landscape_angle,
                 funname=edgeset.method.funname,
                 edges_created_by=creator,
                 iterations=iterations)

    G.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                      for n in range(T)))
    G.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                      for r in range(-R, 0)))

    D = edgeset.D or 0
    if D > 0:
        G.graph['D'] = D
        detournodes = range(T, T + D)
        G.add_nodes_from(((s, {'kind': 'detour'})
                          for s in detournodes))
        clone2prime = edgeset.clone2prime
        assert len(clone2prime) == D, \
            'len(EdgeSet.clone2prime) != EdgeSet.D'
        fnT = np.arange(T + D + R)
        fnT[T: T + D] = clone2prime
        fnT[-R:] = range(-R, 0)
        G.graph['fnT'] = fnT
        AllnodesC = np.vstack((VertexC[:T],
                               VertexC[clone2prime],
                               VertexC[-R:]))
    else:
        AllnodesC = VertexC

    edges = pickle.loads(edgeset.edges)
    U, V = edges.T
    Length = np.hypot(*(AllnodesC[U] - AllnodesC[V]).T)
    G.add_weighted_edges_from(zip(U, V, Length), weight='length')
    if D > 0:
        for _, _, edgeD in G.edges(detournodes, data=True):
            edgeD['kind'] = 'detour'
    calc_length = G.size(weight='length')
    assert abs(calc_length - edgeset.length) < 1, (
        f'{calc_length} != {edgeset.length}')

    if edgeset.method.options is not None:
        G.graph['creation_options'] = json.loads(edgeset.method.options)
    calcload(G)
    return G


def packnodes(G):
    R = G.graph['R']
    T = G.number_of_nodes() - R
    VertexCpkl = pickle.dumps(np.round(G.graph['VertexC'], 2))
    boundarypkl = pickle.dumps(np.round(G.graph['boundary'], 2))
    digest = sha256(VertexCpkl + boundarypkl).digest()
    pack = dict(
        digest=digest,
        name=G.name,
        T=T,
        R=R,
        VertexC=VertexCpkl,
        boundary=boundarypkl,
        landscape_angle=G.graph.get('landscape_angle', 0.),
    )
    return pack


def packmethod(fun, options=None):
    options = options or {}
    if 'capacity' in options:
        del options['capacity']
    funhash = sha256(fun.__code__.co_code).digest()
    optionsstr = json.dumps(options)
    digest = sha256(funhash + optionsstr.encode()).digest()
    pack = dict(
        digest=digest,
        funname=fun.__name__,
        funhash=funhash,
        options=optionsstr,
    )
    return pack


def add_if_absent(entity, pack):
    digest = pack['digest']
    with db_session:
        if not entity.exists(digest=digest):
            entity(**pack)
    return digest


def method_from_graph(G, db):
    '''Returns primary key of the entry.'''
    pack = packmethod(G.graph['edges_fun'], G.graph['creation_options'])
    return add_if_absent(db.Method, pack)


def nodeset_from_graph(G, db):
    '''Returns primary key of the entry.'''
    pack = packnodes(G)
    return add_if_absent(db.NodeSet, pack)


def edgeset_from_graph(G, db):
    '''Adds a new EdgeSet entry in the database, using the data in `G`.
    If the NodeSet or Method are not yet in the database, they will be added.
    '''
    misc_not = {'VertexC', 'anglesYhp', 'anglesXhp', 'anglesRank', 'angles',
                'd2rootsRank', 'd2roots', 'name', 'boundary', 'capacity',
                'runtime', 'runtime_unit', 'edges_fun', 'D', 'DetourC', 'fnT',
                'crossings', 'landscape_angle'}
    nodesetID = nodeset_from_graph(G, db)
    methodID = method_from_graph(G, db),
    machineID = get_machineID(db)
    R = G.graph['R']
    edgepack = dict(
            edges=pickle.dumps(
                np.array([((u, v) if u < v else (v, u))
                          for u, v in G.edges])),
            length=G.size(weight='length'),
            gates=[len(G[root]) for root in range(-R, 0)],
            capacity=G.graph['capacity'],
            runtime=G.graph['runtime'],
            runtime_unit=G.graph['runtime_unit'],
            misc=pickle.dumps({key: G.graph[key]
                               for key in G.graph.keys() - misc_not}),
    )

    D = G.graph.get('D')
    if D is not None and D > 0:
        T_plus_D = G.number_of_nodes() - R
        assert len(G.graph['fnT']) == T_plus_D + R, \
            "len(fnT) != T + D + R"
        edgepack['D'] = D
        edgepack['clone2prime'] = G.graph['fnT'][T_plus_D - D: T_plus_D]
    else:
        edgepack['D'] = 0
    with db_session:
        edgepack.update(
            nodes=db.NodeSet[nodesetID],
            method=db.Method[methodID],
            machine=db.Machine[machineID],
        )
        db.EdgeSet(**edgepack)


def get_machineID(db):
    fqdn = getfqdn()
    hostname = gethostname()
    if fqdn == 'localhost':
        machine = hostname
    else:
        if hostname.startswith('n-'):
            machine = fqdn[len(hostname):]
        else:
            machine = fqdn
    with db_session:
        if not db.Machine.exists(name=machine):
            newMachine = db.Machine(name=machine).id
            return newMachine
        else:
            oldMachine = db.Machine.get(name=machine).id
            return oldMachine


def G_by_method(G, method, db):
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


def Gs_from_attrs(farm, methods, capacities, db):
    '''Fetch from the database a list of tuples of layouts.
    (each tuple has one G for each of `methods`)
    `farm` must have the desired NodeSet name in the `name` attribute.
    `methods` is a tuple of Method instances.
    `capacities` is an int or sequence thereof.
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
        if len(Gtuple) == 1:
            Gs.append(Gtuple[0])
        else:
            Gs.append(Gtuple)
    if len(Gs) == 1:
        return Gs[0]
    else:
        return Gs
