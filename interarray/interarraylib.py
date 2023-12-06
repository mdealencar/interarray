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


def G_from_TG(T, G_base, capacity=None, load_col=4):
    '''Creates a networkx graph with nodes and data from G_base and edges from
    a T matrix.
    T matrix: [ [u, v, length, load (WT number), cable type], ...]'''
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


def branches_to_nodes(Gsrc: nx.Graph, Asrc: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    G = Gsrc.copy()
    A = Asrc.copy()
    P = A.graph['planar'].copy()
    A.graph['planar'] = P
    diagonals = A.graph['diagonals'].copy()
    A.graph['diagonals'] = diagonals
    d2roots = A.graph['d2roots'].copy()
    A.graph['d2roots'] = d2roots
    M = A.graph['M']
    diag2del = []
    max_load = G.graph['capacity'] if G.graph['rooted'] else 2

    # all edges in G are made Delaunay edges in A
    # all crossings of an edge in G are removed from A
    for u, v in G.edges:
        if not A.has_edge(u, v):
            continue
        # remove the crossings with ⟨u, v⟩
        if A[u][v]['type'] == 'extended':
            # ⟨u, v⟩ is a diagonal
            u_, v_ = (u, v) if u < v else (v, u)
            #  print(f'\next({F[u_]}–{F[v_]}) ', end='')
            t = diagonals[u_, v_]
            s = P[t][u_]['cw']
            # remove the Delaunay edge crossed by ⟨u, v⟩
            A.remove_edge(s, t)
            # remove two other diagonals
            # by examining the two triangles ⟨u, v⟩ belongs to
            triangles = ((s, t, u_), (t, s, v_))
            for a, b, c in triangles:
                d = P[c][b]['cw']
                #  print(F[c], F[b], F[d])
                diag_da = (a, d) if a < d else (d, a)
                if (d == P[b][c]['ccw']) and (diag_da in diagonals):
                    A.remove_edge(*diag_da)
                    #  print(f'del {diag_da}', end='')
                    # del diagonals[diag_da]
                    diag2del.append(diag_da)
                e = P[a][c]['cw']
                #  print(F[a], F[c], F[e])
                diag_eb = (e, b) if e < b else (b, e)
                if (e == P[c][a]['ccw']) and (diag_eb in diagonals):
                    A.remove_edge(*diag_eb)
                    #  print(f'del {diag_eb}', end='')
                    # del diagonals[diag_eb]
                    diag2del.append(diag_eb)
            A[u][v]['type'] = 'delaunay'
            del diagonals[u_, v_]
            # update P, otherwise this swap will have no effect
            P.add_half_edge_cw(u, v, s)
            P.add_half_edge_cw(v, u, t)
            P.remove_edge(s, t)
        else:
            # ⟨u, v⟩ is a Delaunay edge -> remove diagonal
            s = P[u][v]['ccw']
            t = P[v][u]['ccw']
            if (t == P[u][v]['cw']
                    and s == P[v][u]['cw']
                    and A.has_edge(s, t)):
                A.remove_edge(s, t)
                diag2del.append((s, t) if s < t else (t, s))
                #  print(f' -{F[s]}–{F[t]}', end='')
    #  print()

    # merge together the connected nodes in G
    G_nodes = nx.subgraph_view(G, filter_node=lambda n: n >= 0)
    #  leaves = tuple(n for n in G.nodes if G.degree[n] == 1)
    leaves = {n for n in G.nodes if G.degree[n] == 1}
    retrace = {}
    for level in range(1, max_load):
        print(f'level: {level}')
        next_leaves = []
        #  for v in leaves:
        while leaves:
            v = leaves.pop()
            print(f'v: {F[v]}', end=', ')
            # merge node v to node u (u inherits v's neighbors)
            u, = G[v]
            print(f'u: {F[u]}', end=', ')
            if u < 0:
                # skip if u is a root
                continue
            if G.degree[u] == 1 and u in leaves:
                # this is necessary for groupings that are not rooted
                leaves.remove(u)
            # print(f'{F[u]}–{F[v]} ({A[u][v]["type"][:3]}):', end='')
            #  print(f'{F[u]}–{F[v]}:', end='')

            A.remove_edge(u, v)
            nbr_u = list(A.neighbors(u))
            # print(f' nbr_u: {nbr_u}', end='')
            for nbv in A.neighbors(v):
                if nbv in nbr_u:
                    # common neighbor of u and v
                    A[u][nbv]['length'] = min(A[u][nbv]['length'],
                                              A[v][nbv]['length'])
                    if A[v][nbv]['type'] == 'delaunay':
                        A[u][nbv]['type'] = 'delaunay'
                        s, t = (u, nbv) if u < nbv else (nbv, u)
                        if (s, t) in diagonals:
                            print(f'deleting diagonal {F[s]}–{F[t]}')
                            del diagonals[s, t]
                else:
                    # neighbor of v but not of u
                    #  print(f' +{F[u]}–{F[nbv]}', end='')
                    A.add_edge(u, nbv, **A[v][nbv])
                    # TODO: add to P
                s, t = (v, nbv) if v < nbv else (nbv, v)
                if (s, t) in diagonals:
                    del diagonals[s, t]
                #  if A[v][nbv]['type'] == 'extended':
                #      del diagonals[(v, nbv) if v < nbv else (nbv, v)]
            merged = A.nodes[u].get('merged', [])
            merged.append((v, A.nodes[v].get('merged', [])))
            #  print(' ', merged, end='')
            # merged.extend(A.nodes[v].get('merged', ()))
            A.nodes[u]['merged'] = merged
            G.remove_node(v)
            if G.degree[u] == 1:
                #  print(f' next_leaves <- {F[u]}', end='')
                next_leaves.append(u)
            A.nodes[u]['power'] = (A.nodes[u].get('power', 1)
                                   + A.nodes[v].get('power', 1))
            d2roots[u, -1] = min(d2roots[u, -1], d2roots[v, -1])
            A.remove_node(v)
            retrace[v] = u
            # remove from P
            last = P[v][u]['ccw']
            if last != P[u][v]['cw']:
                last = u
            nbv = P[v][u]['cw']
            ref = P[u][v]['ccw']
            if nbv == ref:
                nbv = P[v][ref]['cw']
                P.remove_edge(v, ref)
            while nbv != last:
                P.add_half_edge_cw(u, nbv, ref)
                P.add_half_edge_cw(nbv, u, v)
                ref = nbv
                nbv = P[v][ref]['cw']
                P.remove_edge(v, ref)
            #  print(f'removing from P: {F[v]}')
            P.remove_node(v)
            # log.append((plotP(A, P), P.copy()))

            #  print()
        leaves = next_leaves
    # remove diag2del from diagonals
    for diag in diag2del:
        if diag in diagonals:
            del diagonals[diag]
    for (s, t), v in diagonals.items():
        while v in retrace:
            v = retrace[v]
        #  if n not in A.nodes:
        #      print(f'node {F[n]} not in A')
        #      del diagonals[s, t]
        #      continue
        diagonals[s, t] = v

    return G, A
