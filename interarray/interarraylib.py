# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import pickle
import sys
import math
from hashlib import sha256

import networkx as nx
import numpy as np

from .utils import NodeTagger
from .geometric import rotate

F = NodeTagger()

_essential_graph_attrs = (
    'R', 'T', 'B', 'VertexC', 'name', 'handle', 'border',  # required
    'obstacles', 'landscape_angle',  # optional
    'norm_scale', 'norm_offset',  # optional
)


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
        dist += np.hypot(*(VertexC[p] - VertexC[n]).T).item()
        p = n
    return dist


def count_diagonals(S: nx.Graph, A: nx.Graph) -> int:
    '''Count the number of Delaunay diagonals (extended edges) of `A` in `S`.

    Args:
        S: solution topology
        A: available edges used in creating `S`

    Returns:
        number of non-gate edges of `S` that are of kind 'extended' or
            'contour_extended' (kind is read from `A`).

    Raises:
        ValueError: if an edge of unknown kind is found.
    '''
    delaunay = 0
    extended = 0
    gates = 0
    other = 0
    for u, v in S.edges:
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
    assert S.number_of_edges() == delaunay + extended + gates + other
    return extended


def bfs_subtree_loads(G, parent, children, subtree):
    '''
    Recurse down the subtree, updating edge and node attributes. Return value
    is total descendant nodes. Meant to be called by `calcload()`, but can be
    used independently (e.g. from PathFinder).
    Nodes must not have a 'load' attribute.
    '''
    T = G.graph['T']
    nodeD = G.nodes[parent]
    default = 1 if parent < T else 0  # load is 1 for wtg nodes
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
    R, T = (G.graph[k] for k in 'RT')
    roots = range(-R, 0)
    for _, data in G.nodes(data=True):
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
    assert total_load == T, f'counted ({total_load}) != nonrootnodes({T})'
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


def L_from_site(*, VertexC: np.ndarray, T: int, R: int, **kwargs) -> nx.Graph:
    '''
    Args:
        VertexC: numpy.ndarray (V, 2) with x, y pos. of wtg + oss (total V)
        T: int number of wtg
        R: int number of oss
        **kwargs: Additional relevant arguments, for example:
            name: str site name
            handle: str site identifier
            B: int number of border and obstacle zones' vertices
            border: array (B,) of VertexC indices that define the border (ccw)
            obstacles: sequence of numpy.ndarray of VertexC indices

    Returns:
        Graph containing V nodes and no edges. All keyword arguments are made
        available as graph attributes.
    '''
    if 'handle' not in kwargs:
        kwargs['handle'] = 'L_from_site'
    if 'name' not in kwargs:
        kwargs['name'] = kwargs['handle']
    if 'B' not in kwargs:
        border = kwargs.get('border')
        if border is not None:
            kwargs['B'] = border.shape[0]
        else:
            kwargs['B'] = 0
    L = nx.Graph(T=T, R=R,
                 VertexC=VertexC,
                 **kwargs)

    L.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                      for n in range(T)))
    L.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                      for r in range(-R, 0)))
    return L


def G_from_S(S: nx.Graph, A: nx.Graph) -> nx.Graph:
    '''
    Graph `S` contains the topology of a routeset network (nodes only, no
    contours or detours). `S` must have been created from the available edges
    in `A`, whose contour information is used to obtain a routeset `G`
    (possibly with contours, but not with detours – use PathFinder afterward).
    '''
    R, T, B = (A.graph[k] for k in 'RTB')
    VertexC, d2roots, diagonals = (A.graph[k] for k in
                                   ('VertexC', 'd2roots', 'diagonals'))
    # TODO: rethink whether to copy from S or from A
    G = nx.create_empty_copy(S)
    G.graph.update(
        {k: A.graph[k] for k in _essential_graph_attrs + ('num_stunts',)
         if k in A.graph})
    if 'is_normalized' in A.graph:
        G.graph['is_normalized'] = True
    # remove supertriangle coordinates from VertexC
    G.graph['VertexC'] = np.vstack((VertexC[:-R - 3], VertexC[-R:]))
    # non_A_edges are the far-reaching gates and ocasionally the result of
    # a poor solver (e.g. LKH-3)
    non_A_edges = S.edges - A.edges
    # TA_source, TA_target = np.array(S.edges - non_A_edges).T
    common_TA = S.edges - non_A_edges
    iC = T + B
    clone2prime = []
    tentative = []
    shortened_contours = {}
    num_diagonals = 0
    # add to G the S edges that are in A
    for edge in common_TA:
        s, t = edge if edge[0] < edge[1] else edge[::-1]
        AedgeD = A[s][t]
        subtree_id = S.nodes[t]['subtree']
        # only count diagonals that are not gates
        num_diagonals += AedgeD['kind'] == 'extended' and s >= 0
        midpath = AedgeD.get('midpath')

        # This block checks for gate×edge crossings, which may be unnecessary
        # depending on how S was generated. (e.g. creator == 'MILP...' and
        # gateXings_constraint == True).
        st_is_tentative = False
        if s < 0:
            # ⟨s, t⟩ is a gate
            if midpath is not None:
                # While we do not have magic portals, make all contoured gate
                # of kind tentative, so that we do not block access to root
                # around a contour node.
                st_is_tentative = True
            elif (s, t) in diagonals:
                # ⟨s, t⟩ is a diagonal
                u, v = diagonals[(s, t)]
                if (u, v) in S.edges:
                    # ⟨s, t⟩'s Delaunay is in S -> Xing
                    st_is_tentative = True
                else:
                    # check the other diagonals that cross ⟨s, t⟩ (in A)
                    for side in ((u, s), (s, v), (v, t), (t, u)):
                        side = side if side[0] < side[1] else side[::-1]
                        if (side in diagonals.inv
                                and diagonals.inv[side] in S.edges):
                            # side's diagonal is in S -> Xing
                            st_is_tentative = True
                            break
            elif (s, t) in diagonals.inv and diagonals.inv[(s, t)] in S.edges:
                # ⟨s, t⟩ is a Delanay edge and its diagonal is in S -> Xing
                st_is_tentative = True

        load = S[s][t]['load']
        st_reverse = S.nodes[s]['load'] < S.nodes[t]['load']
        if st_is_tentative:
            G.add_edge(s, t, length=AedgeD['length'], load=load,
                       reverse=st_reverse, kind='tentative')
            tentative.append((s, t))
            continue
        if midpath is None:
            # no contour in A's ⟨s, t⟩ -> straightforward
            G.add_edge(s, t, length=AedgeD['length'], load=load,
                       reverse=st_reverse)
            continue
        # contour edge
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
                               length=np.hypot(*(VertexC[s] - VertexC[t]).T)).item()
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
        u = s
        for prime, length in zip(path[1:-1], lengths):
            clone2prime.append(prime)
            v = iC
            iC += 1
            G.add_node(v, kind='contour', load=load, subtree=subtree_id)
            reverse = st_reverse == (u < v)
            G.add_edge(u, v, length=length.item(), load=load, kind='contour',
                       reverse=reverse, A_edge=(s, t))
            u = v
        reverse = st_reverse == (u < t)
        G.add_edge(u, t, length=lengths[-1].item(), load=load, kind='contour',
                   reverse=reverse, A_edge=(s, t))
    if shortened_contours:
        G.graph['shortened_contours'] = shortened_contours
    if clone2prime:
        fnT = np.arange(iC + R)
        fnT[T + B:-R] = clone2prime
        fnT[-R:] = range(-R, 0)
        G.graph.update(fnT=fnT,
                       clone2prime=clone2prime,
                       C=len(clone2prime))
    # add to G the S edges that are not in A
    rogue = []
    for s, t in non_A_edges:
        s, t = (s, t) if s < t else (t, s)
        if s < 0:
            # far-reaching gate
            G.add_edge(s, t, length=d2roots[t, s].item(), kind='tentative',
                       load=S.nodes[t]['load'], reverse=False)
            tentative.append((s, t))
        else:
            # rogue edge (not supposed to be on the routeset, poor solver)
            st_reverse = S.edges[s, t]['reverse']
            load = (S.nodes[s]['load']
                    if st_reverse else
                    S.nodes[t]['load'])
            G.add_edge(s, t, length=np.hypot(*(VertexC[s] - VertexC[t])).item(),
                       kind='rogue', load=load, reverse=st_reverse)
            rogue.append((s, t))
    if rogue:
        G.graph['rogue'] = rogue

    # Check on crossings between G's gates that are in A and G's edges
    diagonals = A.graph['diagonals']
    P = A.graph['planar']
    for r in range(-R, 0):
        for n in set(S.neighbors(r)) & set(A.neighbors(r)):
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
        num_diagonals=num_diagonals,
    )
    return G


def S_from_G(G: nx.Graph):
    '''Get `G`'s topology (contours, detours, lengths, coords are dropped).
    
    If using S to warm-start a MILP model, call after `S_from_G()`:
        - `as_hooked_to_nearest()`: for branching (tree) models
        - `as_hooked_to_head()`: for non-branching (path) models
    This ensures that topology S is feasible (if non-branching) and not
    trivially suboptimal (if branching).

    Args:
        G: must contain a feasible solution (either tree or path)

    Returns:
        topology of `G`
    '''
    R, T = (G.graph[k] for k in 'RT')
    capacity = G.graph['capacity']
    has_loads = G.graph.get('has_loads', False)
    S = nx.Graph(
        T=T, R=R,
        capacity=capacity,
    )
    # create a topology graph S from the results
    for r in range(-R, 0):
        S.add_node(r, kind='oss', **({'load': G.nodes[r]['load']}
                                     if has_loads else {}))
        on_hold = None
        for edge in nx.dfs_edges(G, r):
            u, v = edge
            if v >= T:
                on_hold = u if on_hold is None else on_hold
                continue
            if on_hold is not None:
                u = on_hold
            if has_loads:
                v_load = G.nodes[v]['load']
                S.add_node(v, kind='wtg', load=v_load,
                           subtree=G.nodes[v]['subtree'])
                S.add_edge(u, v, load=G.edges[edge]['load'],
                           reverse=(G.nodes[u]['load'] < v_load) == (u < v))
            else:
                S.add_node(v, kind='wtg')
                S.add_edge(u, v)
            on_hold = None
    creator = G.graph.get('creator')
    if creator is not None:
        S.graph['creator'] = creator
    method_options = G.graph.get('method_options')
    if method_options is not None:
        S.graph['method_options'] = method_options
    if has_loads:
        S.graph['has_loads'] = True
    else:
        calcload(S)
    return S


def L_from_G(G: nx.Graph) -> nx.Graph:
    '''Return new location with nodes and site attributes from G.

    The returned location graph `L` retains only roots, nodes and basic graph
    attributes. All edges and remaining attributes are not carried from `G`.

    Args:
        G: routeset graph to extract site data from.

    Returns:
        Site graph (no edges) with lean attributes.
    '''
    R, T = (G.graph[k] for k in 'RT')
    L = nx.Graph(**{k: G.graph[k]
                    for k in _essential_graph_attrs if k in G.graph})
    num_stunts = G.graph.get('num_stunts')
    if num_stunts:
        VertexC = G.graph['VertexC']
        L.graph['VertexC'] = np.vstack((VertexC[:-R - num_stunts],
                                        VertexC[-R:]))
        L.graph['B'] -= num_stunts
    L.add_nodes_from(((n, {'label': label})
                      for n, label in G.nodes(data='label')
                      if 0 <= n < T), kind='wtg')
    for r in range(-R, 0):
        L.add_node(r, label=G.nodes[r].get('label'), kind='oss')
    return L


def as_single_root(Lʹ: nx.Graph) -> nx.Graph:
    '''Make a shallow copy of an instance and reduce its roots to one.

    The output's root is the centroid of the input's roots.

    Args:
        Lʹ: input location

    Returns:
        location with a single root.
    '''
    R, VertexCʹ = (Lʹ.graph[k] for k in ('R', 'VertexC'))
    L = Lʹ.copy()
    if R <= 1:
        return L
    L.remove_nodes_from(range(-R, -1))
    VertexC = VertexCʹ[:-R + 1].copy()
    VertexC[-1] = VertexCʹ[-R:].mean(axis=0)
    L.graph.update(VertexC=VertexC, R=1)
    L.graph['name'] += '.1_OSS'
    L.graph['handle'] += '_1'
    return L


def as_normalized(Aʹ: nx.Graph) -> nx.Graph:
    '''Make a shallow copy of an instance and shift and scale its geometry.

    Coordinates are subtracted by graph attribute 'norm_offset'.
    All lengths and coordinates are multiplied by graph attribute 'norm_scale'.
    Graph attribute 'is_normalized' is set to `True`.
    Affected linear attributes: 'VertexC', 'd2roots' (graph); 'length' (edge).

    Args:
        Aʹ: (or Gʹ) any instance that has inherited 'norm_scale' from an
            edgeset `Aʹ`.

    Returns:
        A copy of the instance with changed coordinates and linear metrics.
    '''
    A = Aʹ.copy()
    norm_factor = A.graph['norm_scale']
    A.graph['is_normalized'] = True
    for _, _, eData in A.edges(data=True):
        eData['length'] *= norm_factor
    A.graph['VertexC'] = norm_factor*(Aʹ.graph['VertexC']
                                      - Aʹ.graph['norm_offset'])
    A.graph['d2roots'] = norm_factor*Aʹ.graph['d2roots']
    return A


def as_rescaled(Gʹ: nx.Graph, L: nx.Graph) -> nx.Graph:
    '''Revert normalization done by `as_normalized()`.

    Args:
        Gʹ: routeset to rescale to pre-normalization size.
        L: (or G or A) locations or routeset to get 'VertexC' from (also
            'd2roots', if available).

    Returns:
        Routeset with coordinates and lengths at site scale.
    '''
    if not Gʹ.graph.get('is_normalized', False):
        # Gʹ is not marked as normalized
        return Gʹ
    G = Gʹ.copy()
    # alternatively, we could do the math, but this safeguards the coord's hash
    G.graph['VertexC'] = L.graph['VertexC']
    denorm_factor = 1/G.graph['norm_scale']
    for _, _, eData in G.edges(data=True):
        eData['length'] *= denorm_factor
    d2roots = L.graph.get('d2roots')
    if d2roots is not None:
        G.graph['d2roots'] = d2roots
    elif 'd2roots' in G.graph:
        del G.graph['d2roots']
    del G.graph['is_normalized']
    # this factor can be used later to scale metadata (such as 'objective')
    G.graph['denormalization'] = denorm_factor
    return G


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
    R, T, B = (G.graph[k] for k in 'RTB')
    VertexC = G.graph['VertexC']
    tentative = []
    for r in range(-R, 0):
        for n in [n for n in G.neighbors(r) if n >= T + B + C]:
            rev = r
            G.remove_edge(n, r)
            while n >= T:
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
        G.graph['fnT'] = np.hstack((fnT[: T + B + C], fnT[-R:]))
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
        G: routeset or topology S
        d2roots: distance from nodes to roots (e.g. A.graph['d2roots'])
    '''
    assert Gʹ.graph.get('has_loads')
    G = Gʹ.copy()
    R, T = G.graph['R'], G.graph['T']
    # mappings to quickly obtain all nodes on a subtree
    num_subtree = sum(G.degree[r] for r in range(-R, 0))
    nodes_from_subtree_id = np.fromiter((list() for _ in range(num_subtree)),
                                        count=num_subtree, dtype=object)
    subtree_from_node = np.empty((T,), dtype=object)
    for n, subtree_id in G.nodes(data='subtree'):
        if 0 <= n < T:
            subtree = nodes_from_subtree_id[subtree_id]
            subtree.append(n)
            subtree_from_node[n] = subtree

    # do the actual rehooking
    # TODO: rehook should take into account the other roots
    #       see PathFinder.create_detours()
    tentative = []
    hook_getter = ((r, nb) for r in range(-R, 0)
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
            # only necessary if using hook_getter (e.g. Gʹ is a S)
            G[r][new_hook]['kind'] = 'tentative'
        tentative.append((r, new_hook))
    G.graph['tentative'] = tentative
    return G


def as_hooked_to_head(Sʹ: nx.Graph, d2roots: np.ndarray) -> nx.Graph:
    '''Only works with solutions where subtrees are paths (radial topology).

    Sifts through all 'tentative' gates' subtrees and re-hook that path to
    the one of its end-nodes that is neares to the respective root according
    to `d2roots`.

    Should be called after `as_undetoured()` if the goal is to use S as a
    warmstart for MILP models.

    Args:
        S: solution topology
        d2roots: distance from nodes to roots (e.g. A.graph['d2roots'])
    '''
    assert Sʹ.graph.get('has_loads')
    S = Sʹ.copy()
    R, T = S.graph['R'], S.graph['T']
    # mappings to quickly obtain all nodes on a subtree
    S_T = nx.subgraph_view(Sʹ, filter_node=lambda n: n >= 0)
    num_subtree = sum(S.degree[r] for r in range(-R, 0))
    nodes_from_subtree_id = np.fromiter((list() for _ in range(num_subtree)),
                                        count=num_subtree, dtype=object)
    subtree_from_node = np.empty((T,), dtype=object)
    headtail_from_subtree_id = np.fromiter(
        (list() for _ in range(num_subtree)), count=num_subtree, dtype=object)
    headtail_from_node = np.empty((T,), dtype=object)
    for n, subtree_id in S.nodes(data='subtree'):
        if 0 <= n < T:
            subtree = nodes_from_subtree_id[subtree_id]
            subtree.append(n)
            subtree_from_node[n] = subtree
            headtail = headtail_from_subtree_id[subtree_id]
            headtail_from_node[n] = headtail
            if S_T.degree[n] <= 1:
                headtail.append(n)

    # do the actual rehooking
    # TODO: rehook should take into account the other roots
    #       see PathFinder.create_detours()
    tentative = []
    hook_getter = ((r, nb) for r in range(-R, 0)
                   for nb in tuple(S.neighbors(r)))
    for r, hook in S.graph.pop('tentative', hook_getter):
        headtail = headtail_from_node[hook]
        new_hook = headtail[np.argmin(d2roots[headtail, r])]
        if new_hook != hook:
            subtree_load = S.nodes[hook]['load']
            S.remove_edge(r, hook)
            S.add_edge(r, new_hook, kind='tentative', load=subtree_load)
            for node in subtree_from_node[hook]:
                del S.nodes[node]['load']

            ref_load = S.nodes[r]['load']
            S.nodes[r]['load'] = ref_load - subtree_load
            total_parent_load = bfs_subtree_loads(S, r, [new_hook],
                                                  S.nodes[new_hook]['subtree'])
            assert total_parent_load == ref_load, \
                f'parent ({total_parent_load}) != expected load ({ref_load})'
        else:
            # only necessary if using hook_getter (e.g. Gʹ is a S)
            S[r][new_hook]['kind'] = 'tentative'
        tentative.append((r, new_hook))
    S.graph['tentative'] = tentative
    return S


def make_remap(G, refG, H, refH):
    '''Create a mapping between two representations of the same site.

    CAUTION: only WTG node remapping is implemented.

    If the nodes in `G` and in `H` represent the same site, but have different
    orientation, scale and node order, the mapping produced here can be used
    with `NetworkX.relabel_nodes(G, remap)` to translate a routeset in G to a
    routeset in H.

    Args:
        G: routeset with obsolete representation.
        refG: two nodes to used as references.
        H: routeset with valid representation.
        refH: two nodes corresponding to `refG`
    '''
    T = G.graph['T']
    VertexC = G.graph['VertexC'][:T]
    vecref = VertexC[refG[1]] - VertexC[refG[0]]
    angleG = np.arctan2(*vecref)
    scaleG = np.hypot(*vecref)
    GvertC = (VertexC - VertexC[refG[0]])/scaleG
    VertexC = H.graph['VertexC'][:T]
    vecref = VertexC[refH[1]] - VertexC[refH[0]]
    angleH = np.arctan2(*vecref)
    scaleH = np.hypot(*vecref)
    HvertC = rotate((VertexC - VertexC[refH[0]])/scaleH,
                    180*(angleH - angleG)/np.pi)
    remap = {}
    for i, coordH in enumerate(HvertC):
        j = np.argmin(np.hypot(*(GvertC - coordH).T))
        remap[j] = i
    return remap


def scaffolded(G: nx.Graph, P: nx.PlanarEmbedding) -> nx.Graph:
    '''Create a new graph merging G and P.

    Useful for visualizing the funnels explored by `pathfinding.PathFinder`.
    `G` must have been created using `P`.

    Args:
      G: network graph for location
      P: planar embedding of location

    Returns:
      Merged graph (pass to `plotting.gplot()` or 'svg.svgplot()`).
    '''
    scaff = P.to_undirected()
    scaff.graph.update(G.graph)
    for attr in 'fnT C'.split():
        if attr in scaff.graph:
            del scaff.graph[attr]
    R, T, B, C, D = (G.graph.get(k, 0) for k in 'R T B C D'.split())
    nx.set_edge_attributes(scaff, 'scaffold', name='kind')
    constraints = P.graph.get('constraint_edges', [])
    for edge in constraints:
        scaff.edges[edge]['kind'] = 'constraint'
    for n, d in scaff.nodes(data=True):
        if n not in G.nodes:
            continue
        d.update(G.nodes[n])
    if C > 0 or D > 0:
        fnT = G.graph['fnT']
    else:
        fnT = np.arange(R + T + B + C + D)
        fnT[-R:] = range(-R, 0)
    for u, v in G.edges:
        st = fnT[u], fnT[v]
        if st in scaff.edges and 'kind' in scaff.edges[st]:
            del scaff.edges[st]['kind']
    VertexC = G.graph['VertexC']
    supertriangleC = P.graph['supertriangleC']
    if G.graph.get('is_normalized'):
        supertriangleC = G.graph['norm_scale']*(supertriangleC
                                                - G.graph['norm_offset'])
    VertexC = np.vstack((VertexC[:-R],
                         supertriangleC,
                         VertexC[-R:]))
    scaff.graph.update(VertexC=VertexC, fnT=fnT)
    if 'capacity' in scaff.graph:
        # hack to prevent `gplot()` from showing infobox
        del scaff.graph['capacity']
    return scaff
