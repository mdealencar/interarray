# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import pickle
import sys
import math
from hashlib import sha256

import networkx as nx
import numpy as np

from .utils import NodeTagger
from .geometric import make_graph_metrics

F = NodeTagger()


def update_lengths(G):
    '''Adds missing edge lengths.
    Changes G in place.'''
    VertexC = G.graph['VertexC']
    for u, v, dataE in G.edges(data=True):
        if 'length' not in dataE:
            dataE['length'] = np.hypot(*(VertexC[u] - VertexC[v]).T)


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


def count_diagonals(T: nx.Graph, A: nx.Graph) -> int:
    '''Count the number of Delaunay diagonals (extended edges) of `A` in `T`.

    Args:
        T: solution topology
        A: available edges used in creating `T`

    Returns:
        number of non-gate edges of `T` that are of kind 'extended' or
            'contour_extended' (kind is read from `A`).

    Raises:
        ValueError: if an edge of unknown kind is found.
    '''
    delaunay = 0
    extended = 0
    gates = 0
    other = 0
    for u, v in T.edges:
        if u < 0 or v < 0:
            gates += 1
            continue
        kind = A[u][v]['kind']
        if kind is not None:
            if kind.endswith('delaunay'):
                delaunay += 1
            elif kind.endswith('extended'):
                extended += 1
            else:
                other += 1
                raise ValueError('Unknown edge kind: ' + kind)
    assert T.number_of_edges() == delaunay + extended + gates + other
    return extended


def bfs_subtree_loads(G, parent, children, subtree):
    '''
    Recurse down the subtree, updating edge and node attributes. Return value
    is total descendant nodes. Meant to be called by `calcload()`, but can be
    used independently (e.g. from PathFinder).
    Nodes must not have a 'load' attribute.
    '''
    N = G.graph['N']
    nodeD = G.nodes[parent]
    default = 1 if parent < N else 0  # load is 1 for wtg nodes
    if not children:
        nodeD['load'] = default
        return default
    load = nodeD.get('load', default)
    for child in children:
        G.nodes[child]['subtree'] = subtree
        grandchildren = set(G[child].keys())
        grandchildren.remove(parent)
        childload = bfs_subtree_loads(G, child, grandchildren, subtree)
        G[parent][child].update(load=childload, reverse=parent > child)
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
    M, N = (G.graph.get(k) for k in ('M', 'N'))
    roots = range(-M, 0)
    for node, data in G.nodes(data=True):
        if 'load' in data:
            del data['load']

    subtree = 0
    total_load = 0
    max_load = 0
    for root in roots:
        G.nodes[root]['load'] = 0
        for subroot in G[root]:
            subtree_load = bfs_subtree_loads(G, root, [subroot], subtree)
            subtree += 1
            max_load = max(max_load, G.nodes[subroot]['load'])
        total_load += G.nodes[root]['load']
    assert total_load == N, f'counted ({total_load}) != nonrootnodes({N})'
    G.graph['has_loads'] = True
    G.graph['max_load'] = max_load


def site_fingerprint(VertexC: np.ndarray, boundary: np.ndarray) \
        -> tuple[bytes, dict[str, bytes]]:
    #  VertexCpkl = pickle.dumps(np.round(VertexC, 2))
    #  boundarypkl = pickle.dumps(np.round(boundary, 2))
    VertexCpkl = pickle.dumps(VertexC)
    boundarypkl = pickle.dumps(boundary)
    return (sha256(VertexCpkl + boundarypkl).digest(),
            dict(VertexC=VertexCpkl, boundary=boundarypkl))


def fun_fingerprint(fun=None) -> dict[str, bytes | str]:
    if fun is None:
        fcode = sys._getframe().f_back.f_code
    else:
        fcode = fun.__code__
    return dict(
            funhash=sha256(fcode.co_code).digest(),
            funfile=fcode.co_filename,
            funname=fcode.co_name,
            )


def G_from_site(*, VertexC: np.ndarray, N: int, M: int, **kwargs) -> nx.Graph:
    '''
    Args:
        VertexC: numpy.ndarray (V, 2) with x, y pos. of wtg + oss (total V)
        N: int number of wtg
        M: int number of oss

    Additional relevant arguments:
    - 'name': str site name
    - 'handle': str site identifier
    - 'B': int number of border and exclusion zones' vertices
    - 'border': numpy.ndarray (B,) of VertexC indices to border vertice coords
    - 'exclusions': sequence of numpy.ndarray of VertexC indices

    Returns:
        Graph containing V nodes and no edges. All keyword arguments are made
        available as graph attributes.
    '''
    if 'handle' not in kwargs:
        kwargs['handle'] = 'G_from_site'
    if 'name' not in kwargs:
        kwargs['name'] = kwargs['handle']
    if 'B' not in kwargs:
        kwargs['B'] = 0
    G = nx.Graph(N=N, M=M,
                 VertexC=VertexC,
                 **kwargs)

    G.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                      for r in range(-M, 0)))
    return G


def G_from_T(T: nx.Graph, A: nx.Graph) -> nx.Graph:
    '''
    Graph `T` contains the topology of a routeset network (nodes only, no
    contours or detours). `T` must have been created from the available edges
    in `A`, whose contour information is used to obtain a routeset `G`
    (possibly with contours, but not with detours – use PathFinder afterward).
    '''
    M, N, B = (A.graph[k] for k in 'MNB')
    VertexC, d2roots = (A.graph[k] for k in ('VertexC', 'd2roots'))
    # TODO: rethink whether to copy from T or from A
    G = nx.create_empty_copy(T)
    G.graph.update({key: A.graph[key] for key in 'B border name handle '
                    'norm_scale norm_offset landscape_angle'.split()})
    if 'is_normalized' in A.graph:
        G.graph['is_normalized'] = True
    # remove supertriangle coordinates from VertexC
    G.graph['VertexC'] = np.vstack((VertexC[:-M - 3], VertexC[-M:]))
    # non_A_edges are the far-reaching gates and ocasionally the result of
    # a poor solver (e.g. LKH-3)
    non_A_edges = T.edges - A.edges
    # TA_source, TA_target = np.array(T.edges - non_A_edges).T
    common_TA = T.edges - non_A_edges
    iC = N + B
    clone2prime = []
    tentative = []
    shortened_contours = {}
    diagonals_used = 0
    # add to G the T edges that are in A
    for edge in common_TA:
        s, t = edge if edge[0] < edge[1] else edge[::-1]
        AedgeD = A[s][t]
        subtree_id = T.nodes[t]['subtree']
        # only count diagonals that are not gates
        diagonals_used += AedgeD['kind'] == 'extended' and s >= 0
        load = T[s][t]['load']
        s_load = T.nodes[s]['load']
        t_load = T.nodes[t]['load']
        st_reverse = s_load < t_load
        midpath = AedgeD.get('midpath')
        if midpath is None:
            # no contour in A's ⟨s, t⟩ -> straightforward
            G.add_edge(s, t, length=AedgeD['length'], load=load,
                       reverse=st_reverse)
            continue
        # contour edge
        u, u_load = s, s_load
        shortcuts = AedgeD.get('shortcuts')
        if shortcuts is not None:
            if len(shortcuts) == len(midpath):
                # contour is a glitch of make_planar_embedding's P_paths
                if s < 0:
                    # ⟨s, t⟩ is a gate -> make it tentative
                    # This is a hack. It will force PathFinder to check for
                    # crossings and the edge will be confirmed a non-A gate.
                    G.add_edge(s, t,
                               kind='tentative', reverse=False, load=load,
                               length=np.hypot(*(VertexC[s] - VertexC[t]).T))
                    tentative.append((s, t))
                    continue
                G.add_edge(s, t,
                           kind='contour', reverse=st_reverse, load=load,
                           length=AedgeD['length'])
                shortened_contours[(s, t)] = midpath, []
                continue
            shortpath = midpath.copy()
            for short in shortcuts:
                shortpath.remove(short)
            shortened_contours[(s, t)] = midpath, shortpath
            midpath = shortpath
        path = [s] + midpath + [t]
        lengths = np.hypot(*(VertexC[path[1:]] - VertexC[path[:-1]]).T)
        for prime, length in zip(path[1:-1], lengths):
            clone2prime.append(prime)
            if prime not in G.nodes:
                G.add_node(prime)
            v = iC
            iC += 1
            clones = G.nodes[prime].get('clones')
            if clones is None:
                clones = [v]
            else:
                clones.append(v)
            G.add_node(v, kind='contour', load=load, subtree=subtree_id)
            reverse = st_reverse == (u < v)
            G.add_edge(u, v, length=length, load=load, kind='contour',
                       reverse=reverse, A_edge=(s, t))
            u = v
        reverse = st_reverse == (u < t)
        G.add_edge(u, t, length=lengths[-1], load=load, kind='contour',
                   reverse=reverse, A_edge=(s, t))
    if shortened_contours:
        G.graph['shortened_contours'] = shortened_contours
    if clone2prime:
        fnT = np.arange(iC + M)
        fnT[N + B:-M] = clone2prime
        fnT[-M:] = range(-M, 0)
        G.graph.update(fnT=fnT,
                       clone2prime=clone2prime,
                       C=len(clone2prime))
    # add to G the T edges that are not in A
    rogue = []
    for s, t in non_A_edges:
        s, t = (s, t) if s < t else (t, s)
        if s < 0:
            # far-reaching gate
            G.add_edge(s, t, length=d2roots[t, s], kind='tentative',
                       load=T.nodes[t]['load'], reverse=False)
            tentative.append((s, t))
        else:
            # rogue edge (not supposed to be on the routeset, poor solver)
            st_reverse = T.edges[s, t]['reverse']
            load = (T.nodes[s]['load']
                    if st_reverse else
                    T.nodes[t]['load'])
            G.add_edge(s, t, length=np.hypot(*(VertexC[s] - VertexC[t])),
                       kind='rogue', load=load, reverse=st_reverse)
            rogue.append((s, t))
    if rogue:
        G.graph['rogue'] = rogue

    # Check on crossings between G's gates that are in A and G's edges
    diagonals = A.graph['diagonals']
    P = A.graph['planar']
    for r in range(-M, 0):
        for n in set(T.neighbors(r)) & set(A.neighbors(r)):
            #  TODO: if ⟨r, n⟩ is a contour in A, G[r][n] might fail. FIXIT
            st = diagonals.get((r, n))
            if st is not None:
                # st is a Delaunay edge
                if st in G.edges:
                    G[r][n]['kind'] = 'tentative'
                    tentative.append((r, n))
                    continue
                crossings = False
                s, t = st
                # ensure u–s–v–t is ccw
                u, v = ((r, n)
                        if (P[r][t]['cw'] == s and P[n][s]['cw'] == t) else
                        (n, r))
                # examine the two triangles ⟨s, t⟩ belongs to
                for a, b, c in ((s, t, u), (t, s, v)):
                    # this is for diagonals crossing diagonals
                    d = P[c][b]['ccw']
                    diag_da = (a, d) if a < d else (d, a)
                    if (d == P[b][c]['cw'] and diag_da in G.edges):
                        crossings = True
                        break
                    e = P[a][c]['ccw']
                    diag_eb = (e, b) if e < b else (b, e)
                    if (e == P[c][a]['cw'] and diag_eb in G.edges):
                        crossings = True
                        break
                if crossings:
                    G[r][n]['kind'] = 'tentative'
                    tentative.append((r, n))
                    continue
            else:
                uv = diagonals.inv.get((r, n))
                if uv is not None and uv in G.edges:
                    # uv is a Delaunay edge crossing ⟨r, n⟩
                    G[r][n]['kind'] = 'tentative'
                    tentative.append((r, n))
                    continue
    if tentative:
        G.graph['tentative'] = tentative

    G.graph.update(
        diagonals_used=diagonals_used,
        overfed=[len(G[r])/math.ceil(N/T.graph['capacity'])*M
                 for r in range(-M, 0)],
    )
    return G


def T_from_G(G: nx.Graph):
    M, N, B = (G.graph[k] for k in 'MNB')
    fnT, capacity = (G.graph.get(k) for k in ('fnT', 'capacity'))
    has_loads = G.graph.get('has_loads', False)
    T = nx.Graph(
        N=N, M=M,
        capacity=capacity,
    )
    # create a topology graph T from the results
    for r in range(-M, 0):
        T.add_node(r, kind='oss', **({'load': G.nodes[r]['load']}
                                     if has_loads else {}))
        on_hold = None
        for edge in nx.dfs_edges(G, r):
            u, v = edge
            if v >= N:
                on_hold = on_hold or u
                continue
            u = on_hold or u
            if has_loads:
                v_load = G.nodes[v]['load']
                T.add_node(v, kind='wtg', load=v_load,
                           subtree=G.nodes[v]['subtree'])
                T.add_edge(u, v, load=G.edges[edge]['load'],
                           reverse=(G.nodes[u]['load'] < v_load) == (u < v))
            else:
                T.add_node(v, kind='wtg')
                T.add_edge(u, v)
            on_hold = None
    if has_loads:
        T.graph['has_loads'] = True
    else:
        calcload(T)
    return T


def S_from_G(G: nx.Graph) -> nx.Graph:
    '''Return new graph with nodes and site attributes from G.

    The returned site graph `S` retains only roots, nodes and site graph
    attributes. All edges and remaining data are not carried from `G`.

    Args:
        G: routeset graph to extract site data from.

    Returns:
        Site graph (no edges) with lean attributes.
    '''
    M, N, B = (G.graph[k] for k in 'MNB')
    transfer_fields = ('name', 'handle', 'VertexC', 'N', 'M', 'B', 'border',
                       'exclusions', 'landscape_angle')
    S = nx.Graph(**{k: G.graph[k] for k in transfer_fields if k in G.graph})
    S.add_nodes_from(((n, {'label': label})
                      for n, label in G.nodes(data='label')
                      if 0 <= n < N), kind='wtg')
    for r in range(-M, 0):
        S.add_node(r, label=G.nodes[r]['label'], kind='oss')
    return S


def as_single_oss(G: nx.Graph) -> nx.Graph:
    '''
    This is redundant with clusterlib.unify_roots().
    '''
    #  But keep this one.
    Gʹ = G.copy()
    M, VertexC = (G.graph.get(k) for k in ('M', 'VertexC'))
    Gʹ.remove_nodes_from(range(-M, -1))
    VertexCʹ = VertexC[:-M + 1].copy()
    VertexCʹ[-1] = VertexC[-M:].mean(axis=0)
    Gʹ.graph.update(VertexC=VertexCʹ, M=1)
    Gʹ.graph['name'] += '.1_OSS'
    Gʹ.graph['handle'] += '_1'
    return Gʹ


def as_normalized(A: nx.Graph) -> nx.Graph:
    '''Make a shallow copy of an instance and shift and scale its geometry.

    Coordinates are subtracted by graph attribute 'norm_offset'.
    All lengths and coordinates are multiplied by graph attribute 'norm_scale'.
    Graph attribute 'is_normalized' is set to `True`.
    Affected linear attributes: 'VertexC', 'd2roots' (graph); 'length' (edge).

    Args:
        A: any instance that has inherited 'norm_scale' from an edgeset `A`.

    Returns:
        A copy of the instance with changed coordinates and linear metrics.
    '''
    norm_factor = A.graph['norm_scale']
    Aʹ = A.copy()
    Aʹ.graph['is_normalized'] = True
    for u, v, eData in Aʹ.edges(data=True):
        eData['length'] *= norm_factor
    VertexC = norm_factor*(A.graph['VertexC'] - A.graph['norm_offset'])
    Aʹ.graph['VertexC'] = VertexC
    d2roots = norm_factor*A.graph['d2roots']
    Aʹ.graph['d2roots'] = d2roots
    return Aʹ




def as_undetoured(Gʹ: nx.Graph) -> nx.Graph:
    '''
    Create a shallow copy of `Gʹ` without detour nodes (and possibly *with*
    the resulting crossings).

    This is to be applyed to a routeset that already has detours. It serves to
    re-run PathFinder on a detoured routeset, but it is not the best solution
    to prepare a routeset to be used as warmstart (re-hooking is missing).
    '''
    G = Gʹ.copy()
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if not D:
        return G
    M, N, B = (G.graph[k] for k in 'MNB')
    VertexC = G.graph['VertexC']
    tentative = []
    for r in range(-M, 0):
        for n in [n for n in G.neighbors(r) if n >= N + B + C]:
            rev = r
            G.remove_edge(n, r)
            while n >= N:
                rev = n
                n, = G.neighbors(rev)
                G.remove_node(rev)
            G.add_edge(r, n,
                       load=G.nodes[n]['load'],
                       kind='tentative',
                       reverse=False,
                       length=np.hypot(*(VertexC[n] - VertexC[r]).T))
            tentative.append((r, n))
    del G.graph['D']
    if C:
        fnT = G.graph['fnT']
        G.graph['fnT'] = np.hstack((fnT[: N + B + C], fnT[-M:]))
    else:
        del G.graph['fnT']
    G.graph['tentative'] = tentative
    return G


def as_hooked_to_nearest(Gʹ: nx.Graph, d2roots: np.ndarray) -> nx.Graph:
    '''
    Output may be branched (use with care with path routesets).

    Sifts through all 'tentative' gates' subtrees and choose the hook closest
    to the respective root according to `d2roots`.

    Should be called after `as_undetoured()` if the goal is to use G as a
    warmstart for MILP models.

    Args:
        G: routeset or topology T
        d2roots: distance from nodes to roots (e.g. A.graph['d2roots'])
    '''
    G = Gʹ.copy()
    M, N = G.graph['M'], G.graph['N']
    # mappings to quickly obtain all nodes on a subtree
    num_subtree = sum(G.degree[r] for r in range(-M, 0))
    nodes_from_subtree_id = np.fromiter((list() for _ in range(num_subtree)),
                                        count=num_subtree, dtype=object)
    subtree_from_node = np.empty((N,), dtype=object)
    for n, subtree_id in G.nodes(data='subtree'):
        if 0 <= n < N:
            subtree = nodes_from_subtree_id[subtree_id]
            subtree.append(n)
            subtree_from_node[n] = subtree

    # do the actual rehooking
    # TODO: rehook should take into account the other roots
    #       see PathFinder.create_detours()
    tentative = []
    hook_getter = ((r, nb) for r in range(-M, 0)
                   for nb in tuple(G.neighbors(r)))
    for r, hook in G.graph.pop('tentative', hook_getter):
        subtree = subtree_from_node[hook]
        new_hook = subtree[np.argmin(d2roots[subtree, r])]
        if new_hook != hook:
            subtree_load = G.nodes[hook]['load']
            G.remove_edge(r, hook)
            G.add_edge(r, new_hook, length=d2roots[new_hook, r],
                       kind='tentative', load=subtree_load)
            for node in subtree:
                del G.nodes[node]['load']

            ref_load = G.nodes[r]['load']
            G.nodes[r]['load'] = ref_load - subtree_load
            total_parent_load = bfs_subtree_loads(G, r, [new_hook],
                                                  G.nodes[new_hook]['subtree'])
            assert total_parent_load == ref_load, \
                f'parent ({total_parent_load}) != expected load ({ref_load})'
        else:
            # only necessary if using hook_getter (e.g. Gʹ is a T)
            G[r][new_hook]['kind'] = 'tentative'
        tentative.append((r, new_hook))
    G.graph['tentative'] = tentative
    return G


def as_hooked_to_head(Tʹ: nx.Graph, d2roots: np.ndarray) -> nx.Graph:
    '''Only works with solutions where branches are paths.

    Sifts through all 'tentative' gates' subtrees and re-hook that path to
    the one of its end-nodes that is neares to the respective root according
    to `d2roots`.

    Should be called after `as_undetoured()` if the goal is to use T as a
    warmstart for MILP models.

    Args:
        T: solution topology
        d2roots: distance from nodes to roots (e.g. A.graph['d2roots'])
    '''
    T = Tʹ.copy()
    M, N = T.graph['M'], T.graph['N']
    # mappings to quickly obtain all nodes on a subtree
    T_branches = nx.subgraph_view(Tʹ, filter_node=lambda n: n >= 0)
    num_subtree = sum(T.degree[r] for r in range(-M, 0))
    nodes_from_subtree_id = np.fromiter((list() for _ in range(num_subtree)),
                                        count=num_subtree, dtype=object)
    subtree_from_node = np.empty((N,), dtype=object)
    headtail_from_subtree_id = np.fromiter(
        (list() for _ in range(num_subtree)), count=num_subtree, dtype=object)
    headtail_from_node = np.empty((N,), dtype=object)
    for n, subtree_id in T.nodes(data='subtree'):
        if 0 <= n < N:
            subtree = nodes_from_subtree_id[subtree_id]
            subtree.append(n)
            subtree_from_node[n] = subtree
            headtail = headtail_from_subtree_id[subtree_id]
            headtail_from_node[n] = headtail
            if T_branches.degree[n] <= 1:
                headtail.append(n)

    # do the actual rehooking
    # TODO: rehook should take into account the other roots
    #       see PathFinder.create_detours()
    tentative = []
    hook_getter = ((r, nb) for r in range(-M, 0)
                   for nb in tuple(T.neighbors(r)))
    for r, hook in T.graph.pop('tentative', hook_getter):
        headtail = headtail_from_node[hook]
        new_hook = headtail[np.argmin(d2roots[headtail, r])]
        if new_hook != hook:
            subtree_load = T.nodes[hook]['load']
            T.remove_edge(r, hook)
            T.add_edge(r, new_hook, kind='tentative', load=subtree_load)
            for node in subtree_from_node[hook]:
                del T.nodes[node]['load']

            ref_load = T.nodes[r]['load']
            T.nodes[r]['load'] = ref_load - subtree_load
            total_parent_load = bfs_subtree_loads(T, r, [new_hook],
                                                  T.nodes[new_hook]['subtree'])
            assert total_parent_load == ref_load, \
                f'parent ({total_parent_load}) != expected load ({ref_load})'
        else:
            # only necessary if using hook_getter (e.g. Gʹ is a T)
            T[r][new_hook]['kind'] = 'tentative'
        tentative.append((r, new_hook))
    T.graph['tentative'] = tentative
    return T
