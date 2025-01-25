# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
import operator
from collections import defaultdict
from itertools import pairwise, product, combinations
from math import isclose
from typing import Callable

import networkx as nx
import numpy as np
import numba as nb
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .utils import NodeStr, NodeTagger

F = NodeTagger()
NULL = np.iinfo(int).min

def triangle_AR(uC, vC, tC):
    '''returns the aspect ratio of the triangle defined by the three 2D points
    `uC`, `vC` and `tC`, which must be numpy arrays'''
    lengths = np.hypot(*np.column_stack((vC - tC, tC - uC, uC - vC)))
    den = (lengths.sum()/2 - lengths).prod()
    if den == 0.:
        return float('inf')
    else:
        return lengths.prod()/8/den


def any_pairs_opposite_edge(NodesC, uC, vC, margin=0):
    '''Returns True if any two of `NodesC` are on opposite
    sides of the edge (`uC`, `vC`).
    '''
    maxidx = len(NodesC) - 1
    if maxidx <= 0:
        return False
    refC = NodesC[0]
    i = 1
    while point_d2line(refC, uC, vC) <= margin:
        # ref node is approx. overlapping the edge: get the next one
        refC = NodesC[i]
        i += 1
        if i > maxidx:
            return False

    for cmpC in NodesC[i:]:
        if point_d2line(cmpC, uC, vC) <= margin:
            # cmp node is approx. overlapping the edge: skip
            continue
        if not is_same_side(uC, vC, refC, cmpC,
                            touch_is_cross=False):
            return True
    return False


def rotate(coords, angle):
    '''rotates `coords` (numpy array T×2) by `angle` (degrees)'''
    rotation = np.deg2rad(angle)
    c, s = np.cos(rotation), np.sin(rotation)
    return np.dot(coords, np.array([[c, s], [-s, c]]))


def point_d2line(p, u, v):
    '''
    Calculate the distance from point `p` to the line defined by points `u`
    and `v`.
    '''
    x0, y0 = p
    x1, y1 = u
    x2, y2 = v
    return (abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) /
            math.sqrt((x2 - x1)**2 + (y2 - y1)**2))


def angle_numpy(a, pivot, b):
    '''a, pivot, b are coordinate pairs
    returns angle a-root-b (radians)
    - angle is within ±π (shortest arc from a to b around pivot)
    - positive direction is counter-clockwise'''
    A = a - pivot
    B = b - pivot
    # dot_prod = np.dot(A, B) if len(A) >= len(B) else np.dot(B, A)
    dot_prod = A @ B.T  # if len(A) >= len(B) else np.dot(B, A)
    # return np.arctan2(np.cross(A, B), np.dot(A, B))
    return np.arctan2(np.cross(A, B), dot_prod)


def angle(a, pivot, b):
    '''`a`, `pivot`, `b` are coordinate pairs
    returns angle a-root-b (radians)
    - angle is within ±π (shortest arc from a to b around pivot)
    - positive direction is counter-clockwise'''
    Ax, Ay = a - pivot
    Bx, By = b - pivot
    # debug and print(VertexC[a], VertexC[b])
    ang = np.arctan2(Ax*By - Ay*Bx, Ax*Bx + Ay*By)
    # debug and print(f'{ang*180/np.pi:.1f}')
    return ang


def find_edges_bbox_overlaps(
        VertexC: np.ndarray, u: int, v: int, edges: np.ndarray) -> np.ndarray:
    '''Find which `edges` has a bounding box overlap with ⟨u, v⟩.
    
    This is a preliminary filter for crossing checks. Enables avoiding the more
    costly geometric crossing calculations for segments that are clearly
    disjoint.

    Args:
      VertexC: (N×2) point coordinates
      u, v: indices of probed edge
      edges: list of index pairs representing edges to check against
    Returns:
      numpy array with the indices of overlaps in `edges`
    '''
    uC, vC = VertexC[u], VertexC[v]
    edgesC = VertexC[edges]
    return np.flatnonzero(~np.logical_or(
        (edgesC > np.maximum(uC, vC)).all(axis=1),
        (edgesC < np.minimum(uC, vC)).all(axis=1)
    ).any(axis=1))


def is_crossing_numpy(u, v, s, t):
    '''checks if (u, v) crosses (s, t);
    returns ¿? in case of superposition'''

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphics Gems III (http://www.graphicsgems.org/)
    # license: https://github.com/erich666/GraphicsGems/blob/master/LICENSE.md

    A = v - u
    B = s - t

    # bounding box check
    for i in (0, 1):  # X and Y
        lo, hi = (v[i], u[i]) if A[i] < 0 else (u[i], v[i])
        if B[i] > 0:
            if hi < t[i] or s[i] < lo:
                return False
        else:
            if hi < s[i] or t[i] < lo:
                return False

    C = u - s

    # denominator
    f = np.cross(B, A)
    if f == 0:
        # segments are parallel
        return False

    # alpha and beta numerators
    for num in (np.cross(P, Q) for P, Q in ((C, B), (A, C))):
        if f > 0:
            if num < 0 or num > f:
                return False
        else:
            if num > 0 or num < f:
                return False

    # code to calculate intersection coordinates omitted
    # segments do cross
    return True


@nb.njit('f8(f8[:], f8[:])', cache=True, inline='always')
def _cross_prod_2d(P: np.ndarray[tuple[int], np.dtype[np.float64]],
                   Q: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
    return P[0]*Q[1] - P[1]*Q[0]


@nb.njit('b1(f8[:], f8[:], f8[:], f8[:])', cache=True, inline='always')
def is_crossing_no_bbox(uC: np.ndarray[tuple[int], np.dtype[np.float64]],
                        vC: np.ndarray[tuple[int], np.dtype[np.float64]],
                        sC: np.ndarray[tuple[int], np.dtype[np.float64]],
                        tC: np.ndarray[tuple[int], np.dtype[np.float64]]) -> bool:
    '''checks if (uC, vC) crosses (sC, tC);
    returns ¿? in case of superposition
    '''
    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphic Gems III
    A = vC - uC
    B = sC - tC
    C = uC - sC

    # denominator
    #  f = B[0]*A[1] - B[1]*A[0]
    f = _cross_prod_2d(B, A)
    # TODO: arbitrary threshold
    if abs(f) < 1e-10:
        # segments are parallel
        return False

    # alpha and beta numerators
    #  for num in (Px*Qy - Py*Qx for (Px, Py), (Qx, Qy) in ((C, B), (A, C))):
    for P, Q in ((C, B), (A, C)):
        num = _cross_prod_2d(P, Q)
        if f > 0:
            if num < 0 or f < num:
                return False
        else:
            if 0 < num or num < f:
                return False

    # code to calculate intersection coordinates omitted
    # segments do cross
    return True


def is_crossing(uC, vC, sC, tC, touch_is_cross=True):
    '''checks if (uC, vC) crosses (sC, tC);
    returns ¿? in case of superposition
    choices for `less`:
    -> operator.lt counts touching as crossing
    -> operator.le does not count touching as crossing
    '''
    less = operator.lt if touch_is_cross else operator.le

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphic Gems III

    A = vC - uC
    B = sC - tC

    # bounding box check
    for i in (0, 1):  # X and Y
        lo, hi = (vC[i], uC[i]) if A[i] < 0 else (uC[i], vC[i])
        if B[i] > 0:
            if hi < tC[i] or sC[i] < lo:
                return False
        else:
            if hi < sC[i] or tC[i] < lo:
                return False

    Ax, Ay = A
    Bx, By = B
    C = uC - sC

    # denominator
    # print(Ax, Ay, Bx, By)
    f = Bx*Ay - By*Ax
    # print('how close: ', f)
    # TODO: arbitrary threshold
    if isclose(f, 0., abs_tol=1e-10):
        # segments are parallel
        return False

    # alpha and beta numerators
    for num in (Px*Qy - Py*Qx for (Px, Py), (Qx, Qy) in ((C, B), (A, C))):
        if f > 0:
            if less(num, 0) or less(f, num):
                return False
        else:
            if less(0, num) or less(num, f):
                return False

    # code to calculate intersection coordinates omitted
    # segments do cross
    return True


def is_bunch_split_by_corner(bunch, a, o, b, margin=1e-3):
    '''`bunch` is a numpy array of points (T×2)
    the points `a`-`o`-`b` define a corner'''
    AngleA = angle_numpy(a, o, bunch)
    AngleB = angle_numpy(b, o, bunch)
    # print('AngleA', AngleA, 'AngleB', AngleB)
    # keep only those that don't fall over the angle-defining lines
    keep = ~np.logical_or(np.isclose(AngleA, 0, atol=margin),
                          np.isclose(AngleB, 0, atol=margin))
    angleAB = angle(a, o, b)
    angAB = angleAB > 0
    inA = AngleA > 0 if angAB else AngleA < 0
    inB = AngleB > 0 if ~angAB else AngleB < 0
    # print(angleAB, keep, inA, inB)
    inside = np.logical_and(keep, np.logical_and(inA, inB))
    outside = np.logical_and(keep, np.logical_or(~inA, ~inB))
    split = any(inside) and any(outside)
    return split, np.flatnonzero(inside), np.flatnonzero(outside)


@nb.njit('b1(f8[:], f8[:], f8[:], f8[:])', cache=True, inline='always')
def is_triangle_pair_a_convex_quadrilateral(
        uC: np.ndarray[tuple[int], np.dtype[np.float64]],
        vC: np.ndarray[tuple[int], np.dtype[np.float64]],
        sC: np.ndarray[tuple[int], np.dtype[np.float64]],
        tC: np.ndarray[tuple[int], np.dtype[np.float64]]) -> bool:
    '''⟨u, v⟩ is the common side;
    ⟨s, t⟩ are the opposing vertices;
    returns False also if it is a triangle
    only works if ⟨s, t⟩ crosses the line defined by ⟨u, v⟩'''
    # this used to be called `is_quadrilateral_convex()`
    # us × ut
    usut = _cross_prod_2d(sC - uC, tC - uC)
    # vt × vs
    vtvs = _cross_prod_2d(tC - vC, sC - vC)
    if usut == 0. or vtvs == 0.:
        # the four vertices form a triangle
        return False
    return (usut > 0.) == (vtvs > 0.)


def is_same_side(L1, L2, A, B, touch_is_cross=True):
    '''Check if points A an B are on the same side
    of the line defined by points L1 and L2.

    Note: often used to check crossings with gate edges,
    where the gate edge A-B is already known to be on a line
    that crosses the edge L1–L2 (using the angle rank).'''

    # greater = operator.gt if touch_is_cross else operator.ge
    greater = operator.ge if touch_is_cross else operator.gt
    # print(L1, L2, A, B)
    (Ax, Ay), (Bx, By), (L1x, L1y), (L2x, L2y) = (A, B, L1, L2)
    denom = (L1x - L2x)
    # test to avoid division by zero
    if denom:
        a = -(L1y - L2y)/denom
        c = -a*L1x - L1y
        num = a*Ax + Ay + c
        den = a*Bx + By + c
        discriminator = num*den
    else:
        # this means the line is vertical (L1x = L2x)
        # which makes the test simpler
        discriminator = (Ax - L1x)*(Bx - L1x)
    return greater(discriminator, 0)


def is_blocking(root, u, v, s, t):
    # s and t are necessarily on opposite sides of uv
    # (because of Delaunay – see the triangles construction)
    # hence, if (root, t) are on the same side, (s, root) are not
    return (is_triangle_pair_a_convex_quadrilateral(u, v, s, root)
            if is_same_side(u, v, root, t)
            else is_triangle_pair_a_convex_quadrilateral(u, v, root, t))


def apply_edge_exemptions(G, allow_edge_deletion=True):
    '''
    should be DEPRECATED (depends on `delaunay_deprecated()`'s triangles)

    exemption is used by weighting functions that take
    into account the angular sector blocked by each edge w.r.t.
    the closest root node
    '''
    E_hull = G.graph['E_hull']
    N_hull = G.graph['N_hull']
    N_inner = set(G.nodes) - N_hull
    R = G.graph['R']
    # T = G.number_of_nodes() - R
    VertexC = G.graph['VertexC']
    # roots = range(T, T + R)
    roots = range(-R, 0)
    triangles = G.graph['triangles']
    angles = G.graph['angles']

    # set hull edges as exempted
    for edge in E_hull:
        G.edges[edge]['exempted'] = True

    # expanded E_hull to contain edges exempted from blockage penalty
    # (edges that do not block line from nodes to root)
    E_hull_exp = E_hull.copy()

    # check if edges touching the hull should be exempted from blockage penalty
    for n_hull in N_hull:
        for n_inner in (N_inner & set([v for _, v in G.edges(n_hull)])):
            uv = frozenset((n_hull, n_inner))
            u, v = uv
            opposites = triangles[uv]
            if len(opposites) == 2:
                s, t = triangles[uv]
                rootC = VertexC[G.edges[u, v]['root']]
                uvstC = tuple((VertexC[n] for n in (*uv, s, t)))
                if not is_blocking(rootC, *uvstC):
                    E_hull_exp.add(uv)
                    G.edges[uv]['exempted'] = True

    # calculate blockage arc for each edge
    zeros = np.full((R,), 0.)
    for u, v, d in list(G.edges(data=True)):
        if (frozenset((u, v)) in E_hull_exp) or (u in roots) or (v in roots):
            angdiff = zeros
        else:
            # angdiff = (angles[:, u] - angles[:, v]) % (2*np.pi)
            # angdiff = abs(angles[:, u] - angles[:, v])
            angdiff = abs(angles[u] - angles[v])
        arc = np.empty((R,), dtype=float)
        for i in range(R):  # TODO: vectorize this loop
            arc[i] = angdiff[i] if angdiff[i] < np.pi else 2*np.pi - angdiff[i]
        d['arc'] = arc
        # if arc is π/2 or more, remove the edge (it's shorter to go to root)
        if allow_edge_deletion and any(arc >= np.pi/2):
            G.remove_edge(u, v)
            print('angles', arc, 'removing «',
                  '–'.join([F[n] for n in (u, v)]), '»')


def perimeter(VertexC, vertices_ordered):
    '''
    `vertices_ordered` represent indices of `VertexC` in clockwise or counter-
    clockwise order.
    '''
    vec = VertexC[vertices_ordered[:-1]] - VertexC[vertices_ordered[1:]]
    return (np.hypot(*vec.T).sum()
            + np.hypot(*(VertexC[vertices_ordered[-1]]
                         - VertexC[vertices_ordered[0]])))


def angle_helpers(L: nx.Graph) -> tuple[np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    '''
    Args:
        L: location (also works with A or G)

    Returns:
        tuple of (angles, anglesRank, anglesXhp, anglesYhp)
    '''

    T, R, VertexC = (L.graph[k] for k in ('T', 'R', 'VertexC'))
    B = L.graph.get('B', 0)
    NodeC = VertexC[:T + B]
    RootC = VertexC[-R:]

    angles = np.empty((T + B, R), dtype=float)
    for n, nodeC in enumerate(NodeC):
        x, y = (nodeC - RootC).T
        angles[n] = np.arctan2(y, x)

    anglesRank = rankdata(angles, method='dense', axis=0)
    # vertex is in the positive-X half-plane
    anglesXhp = abs(angles) < np.pi/2
    # vertex is in the positive-Y half-plane
    anglesYhp = angles >= 0.
    return angles, anglesRank, anglesXhp, anglesYhp


def assign_root(A: nx.Graph) -> None:
    '''Add node attribute 'root' with the root closest to each node.

    Changes A in-place.

    '''
    closest_root = -A.graph['R'] + np.argmin(A.graph['d2roots'], axis=1)
    nx.set_node_attributes(
        A, {n: r.item() for n, r in enumerate(closest_root)}, 'root')


# TODO: get new implementation from Xings.ipynb
# xingsmat, edge_from_Eidx, Eidx__
def get_crossings_map(Edge, VertexC, prune=True):
    crossings = defaultdict(list)
    for i, A in enumerate(Edge[:-1]):
        u, v = A
        uC, vC = VertexC[A]
        for B in Edge[i+1:]:
            s, t = B
            if s == u or t == u or s == v or t == v:
                # <edges have a common node>
                continue
            sC, tC = VertexC[B]
            if is_crossing(uC, vC, sC, tC):
                crossings[frozenset((*A,))].append((*B,))
                crossings[frozenset((*B,))].append((*A,))
    return crossings


# TODO: test this implementation
def complete_graph(G_base: nx.Graph, *, include_roots: bool = False,
                   prune: bool = True, map_crossings: bool = False) \
        -> nx.Graph:
    '''Creates a networkx graph connecting all non-root nodes to every
    other non-root node. Edges with an arc > pi/2 around root are discarded
    The length of each edge is the euclidean distance between its vertices.'''
    R, T = (G_base.graph[k] for k in 'RT')
    VertexC = G_base.graph['VertexC']
    TerminalC = VertexC[:T]
    RootC = VertexC[-R:]
    NodeC = np.vstack((TerminalC, RootC))
    Root = range(-R, 0)
    V = T + (R if include_roots else 0)
    G = nx.complete_graph(V)
    EdgeComplete = np.column_stack(np.triu_indices(V, k=1))
    #  mask = np.zeros((V,), dtype=bool)
    mask = np.zeros_like(EdgeComplete[:, 0], dtype=bool)
    if include_roots:
        # mask root-root edges
        offset = 0
        for i in range(0, R - 1):
            for j in range(0, R - i - 1):
                mask[offset + j] = True
            offset += (V - i - 1)

        # make node indices span -R:(T - 1)
        EdgeComplete -= R
        nx.relabel_nodes(G, dict(zip(range(T, T + R), Root)),
                         copy=False)
        C = cdist(NodeC, NodeC)
    else:
        C = cdist(TerminalC, TerminalC)
    if prune:
        # prune edges that cover more than 90° angle from any root
        SrcC = VertexC[EdgeComplete[:, 0]]
        DstC = VertexC[EdgeComplete[:, 1]]
        for root in Root:
            rootC = VertexC[root]
            # calculates the dot product of vectors representing the
            # nodes of each edge wrt root; then mark the negative ones
            # (angle > pi/2) on `mask`
            mask |= ((SrcC - rootC)*(DstC - rootC)).sum(axis=1) < 0
    Edge = EdgeComplete[~mask]
    # discard masked edges
    G.remove_edges_from(EdgeComplete[mask])
    if map_crossings:
        # get_crossings_map() takes time and space
        G.graph['crossings_map'] = get_crossings_map(Edge, VertexC)
    # assign nodes to roots?
    # remove edges between nodes belonging to distinct roots whose length is
    # greater than both d2root
    G.graph.update(G_base.graph)
    nx.set_node_attributes(G, G_base.nodes)
    for u, v, edgeD in G.edges(data=True):
        edgeD['length'] = C[u, v]
        # assign the edge to the root closest to the edge's middle point
        edgeD['root'] = -R + np.argmin(
            cdist(((VertexC[u] + VertexC[v])/2)[np.newaxis, :], RootC))
    return G


def minimum_spanning_forest(A: nx.Graph) -> nx.Graph:
    '''Create the minimum spanning tree from the Delaunay triangulation in `A`.
    
    If the graph has more than one root, the tree will be split on its longest
    link between each root pair. The output will be a forest instead of a tree.

    '''
    R, T = (A.graph[k] for k in 'RT')
    N = R + T
    P_A = A.graph['planar']
    num_edges= P_A.number_of_edges()
    edges_ = np.empty((num_edges//2, 2), dtype=np.int32)
    length_ = np.empty(edges_.shape[0], dtype=np.float64)
    for i, (u, v) in enumerate((u, v) for u, v in P_A.edges if u < v):
        edges_[i] = u, v
        length_[i] = A[u][v]['length']
    edges_[edges_ < 0] += N
    P_ = coo_array((length_, (*edges_.T,)), shape=(N, N))
    Q_ = scipy_mst(P_)
    U, V = Q_.nonzero()
    U[U >= T] -= N
    V[V >= T] -= N
    S = nx.Graph(T=T, R=R, capacity=T,
        handle=A.graph.get('handle'),
        creator='minimum_spanning_forest',
    )
    for u, v in zip(U, V):
        S.add_edge(u.item(), v.item(), length=Q_[u, v].item())
    if R > 1:
        # if multiple roots, split the MST in multiple trees
        removals = R - 1
        pair_checks = combinations(range(-R, 0), 2)
        paths = []
        while removals:
            if not paths:
                r1, r2 = next(pair_checks)
                try:
                    path = nx.bidirectional_shortest_path(S, r1, r2)
                except nx.NetworkXNoPath:
                    continue
                i = 0
                for j, p in enumerate(path[1:-1], 1):
                    if p < 0:
                        # split path
                        paths.append(path[i:j+1])
                        i = j
                paths.append(path[i:])
            path = paths.pop()
            λ_incumbent = 0.
            uv_incumbent = None
            for u, v, λ_hop in ((u, v, A[u][v]['length']) for u, v in pairwise(path)):
                if λ_hop > λ_incumbent:
                    λ_incumbent = λ_hop
                    uv_incumbent = u, v
            S.remove_edge(*uv_incumbent)
            removals -= 1
    return S


# TODO: MARGIN is ARBITRARY - depends on the scale
def check_crossings(G, debug=False, MARGIN=0.1):
    '''Checks for crossings (touch/overlap is not considered crossing).
    This is an independent check on the tree resulting from the heuristic.
    It is not supposed to be used within the heuristic.
    MARGIN is how far an edge can advance across another one and still not be
    considered a crossing.'''
    VertexC = G.graph['VertexC']
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    raise NotImplementedError('CDT requires changes in this function')
    if C > 0 or D > 0:
        # detournodes = range(T, T + D)
        # G.add_nodes_from(((s, {'kind': 'detour'})
        #                   for s in detournodes))
        # clone2prime = G.graph['clone2prime']
        # assert len(clone2prime) == D, \
        #     'len(clone2prime) != D'
        # fnT = np.arange(T + D + R)
        # fnT[T: T + D] = clone2prime
        # DetourC = VertexC[clone2prime].copy()
        fnT = G.graph['fnT']
        AllnodesC = np.vstack((VertexC[:T], VertexC[fnT[T:T + D]],
                               VertexC[-R:]))
    else:
        fnT = np.arange(T + R)
        AllnodesC = VertexC
    roots = range(-R, 0)
    fnT[-R:] = roots
    n2s = NodeStr(fnT, T)

    crossings = []
    pivot_plus_edge = []

    def check_neighbors(neighbors, w, x, pivots):
        '''Neighbors is a bunch of nodes, `pivots` is used only for reporting.
        (`w`, `x`) is the edge to be checked if it splits neighbors apart.
        '''
        maxidx = len(neighbors) - 1
        if maxidx <= 0:
            return
        ref = neighbors[0]
        i = 1
        while point_d2line(*AllnodesC[[ref, w, x]]) < MARGIN:
            # ref node is approx. overlapping the edge: get the next one
            ref = neighbors[i]
            i += 1
            if i > maxidx:
                return

        for n2test in neighbors[i:]:
            if point_d2line(*AllnodesC[[n2test, w, x]]) < MARGIN:
                # cmp node is approx. overlapping the edge: skip
                continue
            # print(F[fnT[w]], F[fnT[x]], F[fnT[ref]], F[fnT[cmp]])
            if not is_same_side(*AllnodesC[[w, x, ref, n2test]],
                                touch_is_cross=False):
                print(f'ERROR <splitting>: edge {n2s(w, x)} crosses '
                      f'{n2s(ref, *pivots, n2test)}')
                # crossings.append(((w,  x), (ref, pivot, cmp)))
                crossings.append(((w,  x), (ref, n2test)))
                return True

    # TODO: check crossings among edges connected to different roots
    for root in roots:
        # edges = list(nx.edge_dfs(G, source=root))
        edges = list(nx.edge_bfs(G, source=root))
        # outstr = ', '.join([f'«{F[fnT[u]]}–{F[fnT[v]]}»' for u, v in edges])
        # print(outstr)
        potential = []
        for i, (u, v) in enumerate(edges):
            u_, v_ = fnT[u], fnT[v]
            for s, t in edges[(i + 1):]:
                s_, t_ = fnT[s], fnT[t]
                if s_ == u_ or s_ == v_ or t_ == u_ or t_ == v_:
                    # no crossing if the two edges share a vertex
                    continue
                uvst = np.array((u, v, s, t), dtype=int)
                if is_crossing(*AllnodesC[uvst], touch_is_cross=True):
                    potential.append(uvst)
                    distances = np.fromiter(
                        (point_d2line(*AllnodesC[[p, w, x]])
                         for p, w, x in ((u, s, t),
                                         (v, s, t),
                                         (s, u, v),
                                         (t, u, v))),
                        dtype=float,
                        count=4)
                    # print('distances[' +
                    #       ', '.join((F[fnT[n]] for n in (u, v, s, t))) +
                    #       ']: ', distances)
                    nearmask = distances < MARGIN
                    close_count = sum(nearmask)
                    # print('close_count =', close_count)
                    if close_count == 0:
                        # (u, v) crosses (s, t) away from nodes
                        crossings.append(((u, v), (s, t)))
                        # print(distances)
                        print(f'ERROR <edge-edge>: '
                              f'edge «{F[fnT[u]]}–{F[fnT[v]]}» '
                              f'crosses «{F[fnT[s]]}–{F[fnT[t]]}»')
                    elif close_count == 1:
                        # (u, v) and (s, t) touch node-to-edge
                        pivotI, = np.flatnonzero(nearmask)
                        w, x = (u, v) if pivotI > 1 else (s, t)
                        pivot = uvst[pivotI]
                        neighbors = list(G[pivot])
                        entry = (pivot, w, x)
                        if (entry not in pivot_plus_edge and
                                check_neighbors(neighbors, w, x, (pivot,))):
                            pivot_plus_edge.append(entry)
                    elif close_count == 2:
                        # TODO: This case probably never happens, remove it.
                        #       This would only happen for coincident vertices,
                        #       which might have been possible in the past.
                        print('&&&&& close_count = 2 &&&&&')
                        # (u, v) and (s, t) touch node-to-node
                        touch_uv, touch_st = uvst[np.flatnonzero(nearmask)]
                        free_uv, free_st = uvst[np.flatnonzero(~nearmask)]
                        # print(
                        #    f'touch/free u, v :«{F[fnT[touch_uv]]}–'
                        #    f'{F[fnT[free_uv]]}»; s, t:«{F[fnT[touch_st]]}–'
                        #    f'{F[fnT[free_st]]}»')
                        nb_uv, nb_st = list(G[touch_uv]), list(G[touch_st])
                        # print([F[fnT[n]] for n in nb_uv])
                        # print([F[fnT[n]] for n in nb_st])
                        nbNuv, nbNst = len(nb_uv), len(nb_st)
                        if nbNuv == 1 or nbNst == 1:
                            # <a leaf node with a clone – not a crossing>
                            continue
                        elif nbNuv == 2:
                            crossing = is_bunch_split_by_corner(
                                AllnodesC[nb_st],
                                *AllnodesC[[nb_uv[0], touch_uv, nb_uv[1]]],
                                margin=MARGIN)[0]
                        elif nbNst == 2:
                            crossing = is_bunch_split_by_corner(
                                AllnodesC[nb_uv],
                                *AllnodesC[[nb_st[0], touch_st, nb_st[1]]],
                                margin=MARGIN)[0]
                        else:
                            print('UNEXPECTED case!!! Look into it!')
                            # mark as crossing just to make sure it is noticed
                            crossing = True
                        if crossing:
                            print(f'ERROR <split>: edges '
                                  f'«{F[fnT[u]]}–{F[fnT[v]]}» '
                                  f'and «{F[fnT[s]]}–{F[fnT[t]]}» '
                                  f'break a bunch apart at '
                                  f'{F[fnT[touch_uv]]}, {F[fnT[touch_st]]}')
                            crossings.append(((u,  v), (s, t)))
                    else:  # close_count > 2:
                        # segments (u, v) and (s, t) are almost parallel
                        # find the two nodes furthest apart
                        pairs = np.array(((u, v), (u, s), (u, t),
                                          (s, t), (v, t), (v, s)))
                        furthest = np.argmax(
                            np.hypot(*(AllnodesC[pairs[:, 0]] -
                                       AllnodesC[pairs[:, 1]]).T))
                        # print('furthest =', furthest)
                        w, x = pairs[furthest]
                        q, r = pairs[furthest - 3]
                        if furthest % 3 == 0:
                            # (q, r) is contained within (w, x)
                            neighbors = list(G[q]) + list(G[r])
                            neighbors.remove(q)
                            neighbors.remove(r)
                            check_neighbors(neighbors, w, x, (q, r))
                        else:
                            # (u, v) partially overlaps (s, t)
                            neighbors_q = list(G[q])
                            neighbors_q.remove(w)
                            check_neighbors(neighbors_q, s, t, (q,))
                            # print(crossings)
                            neighbors_r = list(G[r])
                            neighbors_r.remove(x)
                            check_neighbors(neighbors_r, u, v, (r,))
                            # print(crossings)
                            if neighbors_q and neighbors_r:
                                for a, b in product(neighbors_q, neighbors_r):
                                    if is_same_side(*AllnodesC[[q, r, a, b]]):
                                        print(f'ERROR <partial overlap>: edge '
                                              f'«{F[fnT[u]]}–{F[fnT[v]]}» '
                                              f'crosses '
                                              f'«{F[fnT[s]]}–{F[fnT[t]]}»')
                                        crossings.append(((u,  v), (s, t)))
    debug and potential and print(
        'potential crossings: ' +
        ', '.join([f'«{F[fnT[u]]}–{F[fnT[v]]}» × «{F[fnT[s]]}–{F[fnT[t]]}»'
                   for u, v, s, t in potential]))
    return crossings


def rotation_checkers_factory(VertexC: np.ndarray) -> tuple[
        Callable[[int, int, int], bool],
        Callable[[int, int, int], bool]]:

    def cw(A: int, B: int, C: int) -> bool:
        """return
            True: if A->B->C traverses the triangle ABC clockwise
            False: otherwise"""
        Ax, Ay = VertexC[A]
        Bx, By = VertexC[B]
        Cx, Cy = VertexC[C]
        return (Bx - Ax) * (Cy - Ay) < (By - Ay) * (Cx - Ax)

    def ccw(A: int, B: int, C: int) -> bool:
        """return
            True: if A->B->C traverses the triangle ABC counter-clockwise
            False: otherwise"""
        Ax, Ay = VertexC[B]
        Bx, By = VertexC[A]
        Cx, Cy = VertexC[C]
        return (Bx - Ax) * (Cy - Ay) < (By - Ay) * (Cx - Ax)

    return cw, ccw


def rotating_calipers(convex_hull: np.ndarray) \
        -> tuple[np.ndarray, float, float, np.ndarray]:
    # inspired by:
    # jhultman/rotating-calipers:
    #   CUDA and Numba implementations of computational geometry algorithms.
    # (https://github.com/jhultman/rotating-calipers)
    """
    argument `convex_hull` is a (H, 2) array of coordinates of the convex hull
        in counter-clockwise order.
    Args:
        convex_hull: (H, 2) array of coordinates of the convex hull
          in counter-clockwise order

    Returns:

    Reference:
        Toussaint, Godfried T. "Solving geometric problems with the rotating
          calipers." Proc. IEEE Melecon. Vol. 83. 1983.
    """
    caliper_angles = np.array([0.5*np.pi, 0, -0.5*np.pi, np.pi], dtype=float)
    area_min = np.inf
    H = convex_hull.shape[0]
    left, bottom = convex_hull.argmin(axis=0)
    right, top = convex_hull.argmax(axis=0)

    calipers = np.array([left, top, right, bottom], dtype=np.int_)

    for _ in range(H):
        # Roll vertices counter-clockwise
        calipers_advanced = (calipers - 1) % H
        # Vectors from previous calipers to candidates
        vec = convex_hull[calipers_advanced] - convex_hull[calipers]
        # Find angles of candidate edgelines
        angles = np.arctan2(vec[:, 1], vec[:, 0])
        # Find candidate angle deltas
        angle_deltas = caliper_angles - angles
        # Select pivot with smallest rotation
        pivot = np.abs(angle_deltas).argmin()
        # Advance selected pivot caliper
        calipers[pivot] = calipers_advanced[pivot]
        # Rotate all supporting lines by angle delta
        caliper_angles -= angle_deltas[pivot]

        # calculate area for current calipers
        angle = caliper_angles[np.abs(caliper_angles).argmin()]
        c, s = np.cos(angle), np.sin(angle)
        calipers_rot = convex_hull[calipers] @ np.array(((c, -s), (s, c)))
        bbox_rot_min = calipers_rot.min(axis=0)
        bbox_rot_max = calipers_rot.max(axis=0)
        area = (bbox_rot_max - bbox_rot_min).prod()
        # check if area is a new minimum
        if area < area_min:
            area_min = area
            best_calipers = calipers.copy()
            best_caliper_angle = angle
            best_bbox_rot_min = bbox_rot_min
            best_bbox_rot_max = bbox_rot_max

    c, s = np.cos(-best_caliper_angle), np.sin(-best_caliper_angle)
    t = best_bbox_rot_max
    b = best_bbox_rot_min
    # calculate bbox coordinates in original reference frame, ccw vertices
    bbox = np.array(((b[0], b[1]),
                     (b[0], t[1]),
                     (t[0], t[1]),
                     (t[0], b[1])),
                    dtype=float) @ np.array(((c, -s), (s, c)))

    return best_calipers, best_caliper_angle, area_min, bbox


def normalize_area(G_base: nx.Graph, *, hull_nonroot: np.ndarray) -> nx.Graph:
    """
    DEPRECATED: use interarraylib.as_normalized()

    Rescale graph's coordinates and distances such as to make the rootless
    concave hull of nodes enclose an area of 1.
    Graph is first rotated by attribute 'landscape_angle' and afterward it's
    coordinates are translated to the first quadrant, touching the axes.
    The last step is the scaling.
    Graph attributes added/changed:
        'angle': original landscape_angle value
        'offset': values subtracted from coordinates (x, y) before scaling
        'scale': multiplicative factor applied to coordinates
        'landscape_angle': set to 0
    """
    G = nx.create_empty_copy(G_base)
    landscape_angle = G.graph.get('landscape_angle')
    VertexC = (rotate(G_base.graph['VertexC'], landscape_angle)
               if landscape_angle else
               G_base.graph['VertexC'].copy())
    G.graph['VertexC'] = VertexC
    offsetC = VertexC.min(axis=0)
    scale = 1./np.sqrt(
        area_from_polygon_vertices(*(VertexC[hull_nonroot] - offsetC).T))
    d2roots = G.graph.get('d2roots')
    if d2roots is not None:
        G.graph['d2roots'] = d2roots*scale
    G.graph['scale'] = scale
    G.graph['angle'] = landscape_angle
    G.graph['landscape_angle'] = 0
    G.graph['offset'] = offsetC
    VertexC -= offsetC
    VertexC *= scale
    return G


def denormalize(G_scaled, G_base):
    '''
    DEPRECATED: use interarraylib.as_site_scale()

    note: d2roots will be created in G_base if absent.
    '''
    G = G_scaled.copy()
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC'] = G_base.graph['VertexC']
    fnT = G.graph.get('fnT')
    if fnT is None:
        fnT = np.arange(T + R)
        fnT[-R:] = range(-R, 0)
    d2roots = G_base.graph.get('d2roots')
    if d2roots is None:
        d2roots = cdist(VertexC[:T], VertexC[-R:])
        G_base.graph['d2roots'] = d2roots
    G.graph['d2roots'] = d2roots
    G.graph['landscape_angle'] = G_base.graph['landscape_angle']
    ulength = G.graph.get('undetoured_length')
    if ulength is not None:
        G.graph['undetoured_length'] = ulength/G.graph['scale']
    for key in ('angle', 'scale', 'offset'):
        del G.graph[key]
    # TODO: change this to the vectorized version
    for u, v, edgeD in G.edges(data=True):
        edgeD['length'] = np.hypot(*(VertexC[fnT[u]] - VertexC[fnT[v]]).T)
    return G


def area_from_polygon_vertices(X: np.ndarray, Y: np.ndarray) -> float:
    '''Calculate the area enclosed by the polygon with the vertices (x, y).

    Vertices must be in sequence around the perimeter (either clockwise or
    counter-clockwise).

    Args:
        X: array of X coordinates
        Y: array of Y coordinates
    Returns:
        area
    '''
    # Shoelace formula for area (https://stackoverflow.com/a/30408825/287217).
    return 0.5*abs(X[-1]*Y[0] - Y[-1]*X[0]
                   + np.dot(X[:-1], Y[1:])
                   - np.dot(Y[:-1], X[1:]))


@nb.njit(nb.int_(nb.int_[:], nb.int_), cache=True, inline='always')
def index(array: np.ndarray[tuple[int], np.dtype[np.int_]], item: np.int_) -> int:
    for idx, val in enumerate(array):
        if val == item:
            return idx
    # value not found (must not happen, maybe should throw exception)
    # raise ValueError('value not found in array')
    return 0


@nb.njit('void(int_[:, ::1], int_[:, ::1], int_[:, ::1], boolean[:])', cache=True)
def halfedges_from_triangulation(
       triangles: np.ndarray[tuple[int, int], np.dtype[np.int_]],
       neighbors: np.ndarray[tuple[int, int], np.dtype[np.int_]],
       halfedges: np.ndarray[tuple[int, int], np.dtype[np.int_]],
       ref_is_cw_: np.ndarray[tuple[int], np.dtype[np.bool_]]) -> None:
    '''
    Meant to be called from `mesh.planar_from_cdt_triangles()`. Inputs are
    derived from `PythonCDT.Triangulation().triangles`.

    Args:
        triangles: array of triangle.vertices for triangle in triangles
        neighbors: array of triangle.neighbors for triangle in triangles

    Returns:
        3 lists of half-edges to be passed to `networkx.PlanarEmbedding`
    '''
    NULL_ = nb.int_(NULL)
    nodes_done = set()
    # add the first three nodes to process
    nodes_todo = {n: nb.int_(0) for n in triangles[0]}
    i = nb.int_(0)
    while nodes_todo:
        pivot, tri_idx_start = nodes_todo.popitem()
        tri = triangles[tri_idx_start]
        tri_nb = neighbors[tri_idx_start]
        pivot_idx = index(tri, pivot)
        succ_start = tri[(pivot_idx + 1) % 3]
        nb_idx_start_reverse = (pivot_idx - 1) % 3
        succ_end = tri[(pivot_idx - 1) % 3]
        # first half-edge from `pivot`
        #  print('INIT', [pivot, succ_start])
        halfedges[i] = pivot, succ_start, NULL_
        i += 1
        nb_idx = pivot_idx
        ref = succ_start
        ref_is_cw = False
        while True:
            tri_idx = tri_nb[nb_idx]
            if tri_idx == NULL_:
                if not ref_is_cw:
                    # revert direction
                    ref_is_cw = True
                    #  print('REVE', [pivot, succ_end, ref], cw)
                    ref_is_cw_[i] = ref_is_cw
                    halfedges[i] = pivot, succ_end, succ_start
                    i += 1
                    ref = succ_end
                    tri_nb = neighbors[tri_idx_start]
                    nb_idx = nb_idx_start_reverse
                    continue
                else:
                    break
            tri = triangles[tri_idx]
            tri_nb = neighbors[tri_idx]
            pivot_idx = index(tri, pivot)
            succ = (tri[(pivot_idx - 1) % 3]
                    if ref_is_cw else
                    tri[(pivot_idx + 1) % 3])
            nb_idx = ((pivot_idx - 1) % 3) if ref_is_cw else pivot_idx
            #  print('NORM', [pivot, succ, ref], cw)
            ref_is_cw_[i] = ref_is_cw
            halfedges[i] = pivot, succ, ref
            i += 1
            if succ not in nodes_todo and succ not in nodes_done:
                nodes_todo[succ] = tri_idx
            if succ == succ_end:
                break
            ref = succ
        nodes_done.add(pivot)
    return
