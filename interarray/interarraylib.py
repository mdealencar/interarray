# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import pickle
import sys
from hashlib import sha256
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np

from .utils import NodeTagger
from .geometric import make_graph_metrics

F = NodeTagger()


def new_graph_like(G_base, edges=None):
    '''copies graph and nodes attributes, but not edges'''
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    if edges:
        G.add_edges_from(edges)
    return G


def G_base_from_G(G: nx.Graph) -> nx.Graph:
    '''
    Return new graph with nodes (including label and type) and boundary of G.
    In addition, output graph has metrics.

    Similar to `new_graph_like()`, but works with layout solutions that carry
    a lot of extra info (which it discards).
    '''
    M = G.graph['M']
    N = G.graph['VertexC'].shape[0] - M
    transfer_fields = ('name', 'handle', 'VertexC', 'M', 'boundary',
                       'landscape_angle')
    G_base = nx.Graph(**{k: G.graph[k] for k in transfer_fields})
    G_base.add_nodes_from(((n, {'label': label})
                           for n, label in G.nodes(data='label')
                           if 0 <= n < N), type='wtg')
    for r in range(-M, 0):
        G_base.add_node(r, label=G.nodes[r]['label'], type='oss')
    make_graph_metrics(G_base)
    return G_base


def G_from_site(site: dict) -> nx.Graph:
    VertexC = site['VertexC']
    M = site['M']
    N = len(VertexC) - M
    G = nx.Graph(name=site.get('name', 'unnamed site'),
                 handle=site.get('handle', 'site'),
                 M=M,
                 VertexC=VertexC,
                 boundary=site['boundary'])

    G.add_nodes_from(((n, {'label': F[n], 'type': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'type': 'oss'})
                      for r in range(-M, 0)))
    return G


def G_from_T(T, G_base, capacity=None, cost_scale=1e3):
    '''Creates a networkx graph with nodes and data from G_base and edges from
    a T matrix. (suitable for converting the output of juru's `global_optimizer`)
    T matrix: [ [u, v, length, cable type, load (WT number), cost] ]'''
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    M = G_base.graph['M']
    N = G_base.number_of_nodes() - M

    # indexing differences:
    # T starts at 1, while G starts at -M
    edges = (T[:, :2].astype(int) - M - 1)

    G.add_edges_from(edges)
    nx.set_edge_attributes(
        G, {(int(u), int(v)): dict(length=length, cable=cable, load=load, cost=cost)
            for (u, v), length, (cable, load), cost in
            zip(edges, T[:, 2], T[:, 3:5].astype(int), cost_scale*T[:, 5])})
    G.graph['has_loads'] = True
    G.graph['has_costs'] = True
    G.graph['edges_created_by'] = 'G_from_T()'
    if capacity is not None:
        G.graph['capacity'] = capacity
        G.graph['overfed'] = [len(G[root])/np.ceil(N/capacity)*M
                              for root in range(-M, 0)]
    return G


def G_from_TG(T, G_base, capacity=None, load_col=4):
    '''
    DEPRECATED in favor of `G_from_T()`

    Creates a networkx graph with nodes and data from G_base and edges from
    a T matrix.
    T matrix: [ [u, v, length, load (WT number), cable type], ...]
    '''
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    M = G_base.graph['M']
    N = G_base.number_of_nodes() - M

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
    G.graph['edges_created_by'] = 'G_from_TG()'
    G.graph['prevented_crossings'] = 0
    if capacity is not None:
        G.graph['overfed'] = [len(G[root])/np.ceil(N/capacity)*M
                              for root in range(N, N + M)]
    return G


def update_lengths(G):
    '''Adds missing edge lengths.
    Changes G in place.'''
    VertexC = G.graph['VertexC']
    for u, v, dataE in G.edges.data():
        if 'length' not in dataE:
            dataE['length'] = np.hypot(*(VertexC[u] - VertexC[v]).T)


def bfs_subtree_loads(G, parent, children, subtree):
    '''
    Recurse down the subtree, updating edge and node attributes. Return value
    is total descendant nodes. Meant to be called by `calcload()`, but can be
    used independently (e.g. from PathFinder).
    Nodes must not have a 'load' attribute.
    '''
    nodeD = G.nodes[parent]
    default = 1 if nodeD['type'] == 'wtg' else 0
    if not children:
        nodeD['load'] = default
        return default
    load = nodeD.get('load', default)
    for child in children:
        G.nodes[child]['subtree'] = subtree
        grandchildren = set(G[child].keys())
        grandchildren.remove(parent)
        childload = bfs_subtree_loads(G, child, grandchildren, subtree)
        G[parent][child].update((
            ('load', childload),
            ('reverse', parent > child)
        ))
        load += childload
    nodeD['load'] = load
    return load


def calcload(G):
    '''
    Perform a breadth-first-traversal of each root's subtree. As each node is
    visited, its subtree id and the load leaving it are stored as its
    attribute (keys 'subtree' and 'load', respectively). Also the edges'
    'load' attributes are updated accordingly.
    '''
    M = G.graph['M']
    N = G.number_of_nodes() - M
    D = G.graph.get('D')
    if D is not None:
        N -= D
    roots = range(-M, 0)
    for node, data in G.nodes(data=True):
        if 'load' in data:
            del data['load']

    subtree = 0
    total_load = 0
    for root in roots:
        G.nodes[root]['load'] = 0
        for subroot in G[root]:
            bfs_subtree_loads(G, root, [subroot], subtree)
            subtree += 1
        total_load += G.nodes[root]['load']
    assert total_load == N, f'counted ({total_load}) != nonrootnodes({N})'
    G.graph['has_loads'] = True


def pathdist(G, path):
    '''
    Return the total length (distance) of a `path` of nodes in `G` from nodes'
    coordinates (does not rely on edge attributes).
    '''
    VertexC = G.graph['VertexC']
    dist = 0.
    p = path[0]
    for n in path[1:]:
        dist += np.hypot(*(VertexC[p] - VertexC[n]).T)
        p = n
    return dist


def remove_detours(H: nx.Graph) -> nx.Graph:
    '''
    Create a shallow copy of `H` without detour nodes
    (and *with* the resulting crossings).
    '''
    G = H.copy()
    M = G.graph['M']
    VertexC = G.graph['VertexC']
    N = G.number_of_nodes() - M - G.graph.get('D', 0)
    for r in range(-M, 0):
        detoured = [n for n in G.neighbors(r) if n >= N]
        if detoured:
            G.graph['crossings'] = []
        for n in detoured:
            ref = r
            G.remove_edge(n, r)
            while n >= N:
                ref = n
                n, = G.neighbors(ref)
                G.remove_node(ref)
            G.add_edge(n, r,
                       load=G.nodes[n]['load'],
                       reverse=False,
                       length=np.hypot(*(VertexC[n] - VertexC[r]).T))
            G.graph['crossings'].append((r, n))
    G.graph.pop('D', None)
    G.graph.pop('fnT', None)
    return G


def site_fingerprint(VertexC: np.ndarray, boundary: np.ndarray) \
        -> Tuple[bytes, Dict[str, bytes]]:
    #  VertexCpkl = pickle.dumps(np.round(VertexC, 2))
    #  boundarypkl = pickle.dumps(np.round(boundary, 2))
    VertexCpkl = pickle.dumps(VertexC)
    boundarypkl = pickle.dumps(boundary)
    return (sha256(VertexCpkl + boundarypkl).digest(),
            dict(VertexC=VertexCpkl, boundary=boundarypkl))


def fun_fingerprint(fun=None) -> Dict[str, Any]:
    if fun is None:
        fcode = sys._getframe().f_back.f_code
    else:
        fcode = fun.__code__
    return dict(
            funhash=sha256(fcode.co_code).digest(),
            funfile=fcode.co_filename,
            funname=fcode.co_name,
            )
