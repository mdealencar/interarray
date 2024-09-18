# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import numpy as np
import numpy.lib.recfunctions as nprec
import networkx as nx
from functools import partial

from .heuristics import CPEW, NBEW, OBEW, ClassicEW
from .interarraylib import calcload, F
from .geometric import make_graph_metrics

heuristics = {
    'CPEW': CPEW,
    'NBEW': NBEW,
    'OBEW': OBEW,
    'OBEW_0.6': partial(OBEW, rootlust='0.6*cur_capacity/capacity'),
}


def translate2global_optimizer(G):
    VertexC = G.graph['VertexC']
    M = G.graph['M']
    N = G.graph['N']
    X, Y = np.hstack((VertexC[-1:-1 - M:-1].T, VertexC[:N].T))
    return dict(WTc=N, OSSc=M, X=X, Y=Y)


def assign_cables(G, cables):
    '''
    G: networkx graph with edges having a 'load' attribute (use calcload(G))
    cables: [(«cross section», «capacity», «cost»), ...] in increasing
            capacity order (each cable entry must be a tuple)
            or numpy.ndarray where each row represents one cable type

    The attribute 'cost' of the edges of G will be updated with their cost,
    considering the cheapest cable that can handle their load.
    A new attribute 'cable' will be assigned to each edge with the index of the
    cable chosen.
    '''

    Nc = len(cables)
    dt = np.dtype([('area', float),
                   ('capacity', int),
                   ('cost', float)])
    if isinstance(cables, np.ndarray):
        cable_ = nprec.unstructured_to_structured(cables, dtype=dt)
    else:
        cable_ = np.fromiter(cables, dtype=dt, count=Nc)
    κ_ = cable_['capacity']
    capacity = 1

    # for e, data in G.edges.items():
    for u, v, data in G.edges(data=True):
        i = κ_.searchsorted(data['load'])
        if i >= len(κ_):
            print(f'ERROR: Load for edge ⟨{u, v}⟩: {data["load"]} '
                  f'exceeds maximum cable capacity {κ_[-1]}.')
        data['cable'] = i
        data['cost'] = data['length']*cable_['cost'][i]
        if data['load'] > capacity:
            capacity = data['load']
    G.graph['cables'] = cable_
    G.graph['has_costs'] = True
    if 'capacity' not in G.graph:
        G.graph['capacity'] = capacity


def assign_subtree(G):
    start = 0
    queue = []
    for root in range(-G.graph['M'], 0):
        for subtree, gate in enumerate(G[root], start=start):
            queue.append((root, gate))
            while queue:
                parent, node = queue.pop()
                G.nodes[node]['subtree'] = subtree
                for nbr in G[node]:
                    if nbr != parent:
                        queue.append((node, nbr))
        start = subtree + 1


def G_from_XYM(X, Y, M=1, name='unnamed', borderC=None):
    '''
    This function assumes that the first M vertices are OSSs
    X: x coordinates of vertices
    Y: y coordinates of vertices
    M: number of OSSs
    '''
    assert len(X) == len(Y), 'ERROR: X and Y lengths must match'
    N = len(X) - M

    # create networkx graph
    if borderC is None:
        borderC = np.array((
            (min(X), min(Y)),
            (min(X), max(Y)),
            (max(X), max(Y)),
            (max(X), min(Y))))
    B = borderC.shape[0]
    border = list(range(N, N + B))
    G = nx.Graph(M=M, N=N, B=B, border=border, name=name,
                 VertexC=np.r_[np.c_[X[M:], Y[M:]],
                               np.c_[X[M-1::-1], Y[M-1::-1]],
                               borderC])
    G.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                      for r in range(-M, 0)))
    make_graph_metrics(G)
    return G


def G_from_table(table: np.ndarray[:, :], G_base: nx.Graph,
                 capacity: int | None = None, cost_scale: float = 1e3) \
                 -> nx.Graph:
    '''Creates a networkx graph with nodes and data from G_base and edges from
    a table. (e.g. the T matrix of juru's `global_optimizer`)

    `table`: [ [u, v, length, cable type, load (WT number), cost] ]'''
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    M = G_base.graph['M']
    N = G_base.graph['N']

    # indexing differences:
    # table starts at 1, while G starts at -M
    edges = (table[:, :2].astype(int) - M - 1)

    G.add_edges_from(edges)
    nx.set_edge_attributes(
        G, {(int(u), int(v)):
            dict(length=length, cable=cable, load=load, cost=cost)
            for (u, v), length, (cable, load), cost in
            zip(edges, table[:, 2], table[:, 3:5].astype(int),
                cost_scale*table[:, 5])})
    G.graph['has_loads'] = True
    G.graph['has_costs'] = True
    G.graph['creator'] = 'G_from_table()'
    if capacity is not None:
        G.graph['capacity'] = capacity
        G.graph['overfed'] = [len(G[root])/np.ceil(N/capacity)*M
                              for root in range(-M, 0)]
    return G


def G_from_TG(T, G_base, capacity=None, load_col=4):
    '''
    DEPRECATED in favor of `G_from_table()`

    Creates a networkx graph with nodes and data from G_base and edges from
    a T matrix.
    T matrix: [ [u, v, length, load (WT number), cable type], ...]
    '''
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    M = G_base.graph['M']
    N = G_base.graph['N']

    # indexing differences:
    # T starts at 1, while G starts at 0
    # T begins with OSSs followed by WTGs,
    # while G begins with WTGs followed by OSSs
    # the line bellow converts the indexing:
    edges = (T[:, :2].astype(int) - M - 1) % (N + M)

    G.add_weighted_edges_from(zip(*edges.T, T[:, 2]), weight='length')
    # nx.set_edge_attributes(G, {(u, v): load for (u, v), load
    #                            in zip(edges, T[:, load_col])},
    #                        name='load')
    # try:
    calcload(G)
    # except AssertionError as err:
    #     print(f'>>>>>>>> SOMETHING WENT REALLY WRONG: {err} <<<<<<<<<<<')
    #     return G
    if T.shape[1] >= 4:
        for (u, v), load in zip(edges, T[:, load_col]):
            Gload = G.edges[u, v]['load']
            assert Gload == load, (
                f'<G.edges[{u}, {v}]> {Gload} != {load} <T matrix>')
    G.graph['has_loads'] = True
    G.graph['creator'] = 'G_from_TG()'
    G.graph['prevented_crossings'] = 0
    if capacity is not None:
        G.graph['overfed'] = [len(G[root])/np.ceil(N/capacity)*M
                              for root in range(N, N + M)]
    return G


def table_from_G(G):
    '''
    G: networkx graph

    returns:
    table: [ («u», «v», «length», «load (WT number)», «cable type»,
              «edge cost»), ...]

    (table is a numpy record array)
    '''
    M = G.graph['M']
    Ne = G.number_of_edges()

    def edge_parser(edges):
        for u, v, data in edges:
            # OSS index starts at 0
            # u = (u + M) if u > 0 else abs(u) - 1
            # v = (v + M) if v > 0 else abs(v) - 1
            # OSS index starts at 1
            s = (u + M + 1) if u >= 0 else abs(u)
            t = (v + M + 1) if v >= 0 else abs(v)
            # print(u, v, '->', s, t)
            yield (s, t, data['length'], data['load'], data['cable'],
                   data['cost'])

    table = np.fromiter(edge_parser(G.edges(data=True)),
                        dtype=[('u', int),
                               ('v', int),
                               ('length', float),
                               ('load', int),
                               ('cable', int),
                               ('cost', float)],
                        count=Ne)
    return table


class HeuristicFactory():
    '''
    Initializes a heuristic algorithm.
    Inputs:
    N: number of nodes
    M: number of roots
    rootC: 2D nympy array (M, 2) of the XY coordinates of the roots
    boundaryC: 2D numpy array (_, 2) of the XY coordinates of the boundary
    cables: [(«cross section», «capacity», «cost»), ...] ordered by capacity
    name: site name

    (increasing capacity along cables' elements)
    '''

    def __init__(self, N, M, rootC, boundaryC, heuristic, cables,
                 name='unnamed'):
        self.N = N
        self.M = M
        self.cables = cables
        self.k = cables[-1][1]
        self.VertexC = np.empty((N + M, 2), dtype=float)
        self.VertexC[N:] = rootC
        # create networkx graph
        self.G_base = nx.Graph(M=M,
                               VertexC=self.VertexC,
                               boundary=boundaryC,
                               name=name)
        self.G_base.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                                    for n in range(N)))
        self.G_base.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                                    for r in range(-M, 0)))
        self.heuristic = heuristics[heuristic]

    def calccost(self, X, Y):
        assert len(X) == len(Y) == self.N
        self.VertexC[:self.N, 0] = X
        self.VertexC[:self.N, 1] = Y
        make_graph_metrics(self.G_base)
        self.G = self.heuristic(self.G_base, capacity=self.k)
        calcload(self.G)
        assign_cables(self.G, self.cables)
        return self.G.size(weight='cost')

    def get_table(self):
        '''
        Must have called cost() at least once. Only the last call's layout is
        available.
        returns:
        table: [ («u», «v», «length», «load (WT number)», «cable type»,
                  «edge cost»), ...]
        '''
        return table_from_G(self.G)


def heuristic_wrapper(X, Y, cables, M=1, heuristic='CPEW', return_graph=False):
    '''
    This function assumes that the first M vertices are OSSs
    X: x coordinates of vertices
    Y: y coordinates of vertices
    cables: [(«cross section», «capacity», «cost»), ...] ordered by capacity
    M: number of OSSs
    heuristic: {'CPEW', 'OBEW'}

    (increasing capacity along cables' elements)
    '''
    G_base = G_from_XYM(X, Y, M)
    G = heuristics[heuristic](G_base, capacity=cables[-1][1])
    calcload(G)
    assign_cables(G, cables)
    if return_graph:
        return table_from_G(G), G
    else:
        return table_from_G(G)
