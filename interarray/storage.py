# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import datetime
import json
import pickle
from collections.abc import Sequence
from hashlib import sha256
from socket import getfqdn, gethostname
import networkx as nx
import numpy as np
from pony.orm import db_session, commit

from interarray.dbmodel import NodeSet, EdgeSet, Method, Machine
from interarray.interarraylib import calcload, NodeTagger

# Coordinates use arrays of floats.
# Somehow, nodesets with the same coordinates were getting different digests,
# when the code ran on different computers.
# Rouding to a fixed (small) number of decimal place to fix it.
COORDINATES_DECIMAL_PLACES = 2

F = NodeTagger()


def graph_from_edgeset(edgeset):
    nodeset = edgeset.nodes
    VertexC = pickle.loads(nodeset.VertexC)
    N = nodeset.N
    M = nodeset.M
    G = nx.Graph(name=nodeset.name,
                 M=M,
                 VertexC=VertexC,
                 capacity=edgeset.capacity,
                 boundary=pickle.loads(nodeset.boundary),
                 edges_created_by=edgeset.method.funname)

    G.add_nodes_from(((n, {'label': F[n], 'type': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'type': 'oss'})
                      for r in range(-M, 0)))

    D = edgeset.D
    if D is not None:
        G.graph['D'] = D
        detournodes = range(N, N + D)
        G.add_nodes_from(((s, {'type': 'detour'})
                          for s in detournodes))
        clone2prime = edgeset.clone2prime
        if clone2prime is None:
            # <deprecated format - backwards compatibility>
            DetourC = pickle.loads(edgeset.DetourC)
        else:
            assert len(clone2prime) == D, \
                'len(EdgeSet.clone2prime) != EdgeSet.D'
            fnT = np.arange(N + D + M)
            fnT[N: N + D] = clone2prime
            fnT[-M:] = range(-M, 0)
            G.graph['fnT'] = fnT
            DetourC = VertexC[clone2prime].copy()
        G.graph['DetourC'] = DetourC
        AllnodesC = np.vstack((VertexC[:N], DetourC, VertexC[-M:]))
    else:
        AllnodesC = VertexC

    edges = pickle.loads(edgeset.edges)
    U, V = edges.T
    Length = np.hypot(*(AllnodesC[U] - AllnodesC[V]).T)
    G.add_weighted_edges_from(zip(U, V, Length), weight='length')
    if D is not None:
        for _, _, edgeD in G.edges(detournodes, data=True):
            edgeD['type'] = 'detour'
    G.graph['overfed'] = [len(G[root])/np.ceil(N/edgeset.capacity)*M
                          for root in range(-M, 0)]
    calc_length = G.size(weight='length')
    assert abs(calc_length - edgeset.length) < 1, (
        f'{calc_length} != {edgeset.length}')

    if edgeset.misc is not None:
        miscdict = pickle.loads(edgeset.misc)
        G.graph['iterations'] = miscdict.get('iterations', 1)
    # make_graph_metrics(G)
    calcload(G)
    return G


def edgeset_from_graph(G):
    '''Adds a new EdgeSet entry in the database, using the data in `G`.
    If the NodeSet or Method are not yet in the database, they will be added.
    '''
    misc_not = {'VertexC', 'anglesYhp', 'anglesXhp', 'anglesRank', 'angles',
                'd2rootsRank', 'd2roots', 'name', 'boundary', 'capacity',
                'runtime', 'runtime_unit', 'edges_fun', 'D', 'DetourC', 'fnT',
                'crossings'}
    nodesetID = nodeset_from_graph(G)
    methodID = method_from_graph(G),
    machineID = get_machineID()
    # edges = pickle.dumps(np.array([(u, v) for u, v in G.edges]))
    # length = G.size(weight='length')
    M = G.graph['M']
    # gates = [len(G[root]) for root in range(-M, 0)]
    # cableset
    # misc = pickle.dumps({key: G.graph[key] for key in G.graph.keys() - misc_not})
    edgepack = dict(
            #  edges=pickle.dumps(np.array([(u, v) for u, v in G.edges])),
            edges=pickle.dumps(
                np.array([((u, v) if u < v else (v, u))
                          for u, v in G.edges])),
            length=G.size(weight='length'),
            gates=[len(G[root]) for root in range(-M, 0)],
            capacity=G.graph['capacity'],
            runtime=G.graph['runtime'],
            runtime_unit=G.graph['runtime_unit'],
            misc=pickle.dumps({key: G.graph[key]
                               for key in G.graph.keys() - misc_not}),
    )

    D = G.graph.get('D')
    if D is not None:
        N_plus_D = G.number_of_nodes() - M
        assert len(G.graph['fnT']) == N_plus_D + M, \
            "len(fnT) != N + D + M"
        edgepack['D'] = D
        edgepack['clone2prime'] = G.graph['fnT'][N_plus_D - D: N_plus_D]
        # TODO: deprecate DetourC
        # edgepack['DetourC'] = pickle.dumps(G.graph['DetourC'])
    with db_session:
        edgepack['nodes'] = NodeSet[nodesetID]
        edgepack['method'] = Method[methodID]
        edgepack['machine'] = Machine[machineID]
        EdgeSet(**edgepack)
    commit()


def get_machineID():
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
        if not Machine.exists(name=machine):
            newMachine = Machine(name=machine).id
            commit()
            return newMachine
        else:
            oldMachine = Machine.get(name=machine).id
            return oldMachine


def add_if_absent(entity, pack):
    digest = pack['digest']
    with db_session:
        if not entity.exists(digest=digest):
            entity(**pack)
    return digest


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


def method_from_graph(G):
    '''Returns primary key of the entry.'''
    pack = packmethod(G.graph['edges_fun'], G.graph['creation_options'])
    return add_if_absent(Method, pack)


def packnodes(G):
    M = G.graph['M']
    N = G.number_of_nodes() - M
    VertexCpkl = pickle.dumps(np.round(G.graph['VertexC'], 2))
    boundarypkl = pickle.dumps(np.round(G.graph['boundary'], 2))
    digest = sha256(VertexCpkl + boundarypkl).digest()
    pack = dict(
        digest=digest,
        name=G.name,
        N=N,
        M=M,
        VertexC=VertexCpkl,
        boundary=boundarypkl,
    )
    return pack


def nodeset_from_graph(G):
    '''Returns primary key of the entry.'''
    pack = packnodes(G)
    return add_if_absent(NodeSet, pack)


def G_by_method(G, method):
    '''Fetch from the database a layout for `G` by `method`.
    `G` must be a layout solution with the necessary info in the G.graph dict.
    `method` is a Method.
    '''
    farmname = G.name
    c = G.graph['capacity']
    es = EdgeSet.get(lambda e:
                     e.nodes.name == farmname and
                     e.method is method and
                     e.capacity == c)
    Gdb = graph_from_edgeset(es)
    calcload(Gdb)
    return Gdb


def Gs_from_attrs(farm, methods, capacities):
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
                EdgeSet.get(lambda e:
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
