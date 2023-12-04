# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import functools
import math
import operator
from collections import defaultdict
from itertools import chain, product
from math import isclose

import networkx as nx
import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

from . import MAX_TRIANGLE_ASPECT_RATIO
from .utils import NodeStr, NodeTagger

F = NodeTagger()


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
    '''rotates `coords` (numpy array N×2) by `angle` (degrees)'''
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


def is_bb_overlapping(uv, wy):
    ''' checks if there is an overlap in the bounding boxes of `uv` and `wy`
    (per row)
    `uv` and `wy` have shape N×2, '''
    pass


def is_crossing_numpy(u, v, w, y):
    '''checks if (u, v) crosses (w, y);
    returns ¿? in case of superposition'''

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphics Gems III (http://www.graphicsgems.org/)
    # license: https://github.com/erich666/GraphicsGems/blob/master/LICENSE.md

    A = v - u
    B = w - y

    # bounding box check
    for i in (0, 1):  # X and Y
        lo, hi = (v[i], u[i]) if A[i] < 0 else (u[i], v[i])
        if B[i] > 0:
            if hi < y[i] or w[i] < lo:
                return False
        else:
            if hi < w[i] or y[i] < lo:
                return False

    C = u - w

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


def is_crossing(u, v, w, y, touch_is_cross=True):
    '''checks if (u, v) crosses (w, y);
    returns ¿? in case of superposition
    choices for `less`:
    -> operator.lt counts touching as crossing
    -> operator.le does not count touching as crossing
    '''
    less = operator.lt if touch_is_cross else operator.le

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphic Gems III

    A = v - u
    B = w - y

    # bounding box check
    for i in (0, 1):  # X and Y
        lo, hi = (v[i], u[i]) if A[i] < 0 else (u[i], v[i])
        if B[i] > 0:
            if hi < y[i] or w[i] < lo:
                return False
        else:
            if hi < w[i] or y[i] < lo:
                return False

    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C = u - w

    # denominator
    # print(Ax, Ay, Bx, By)
    f = Bx*Ay - By*Ax
    # print('how close: ', f)
    # TODO: arbitrary threshold
    if isclose(f, 0, abs_tol=1e-3):
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
    '''`bunch` is a numpy array of points (N×2)
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


def is_triangle_pair_a_convex_quadrilateral(u, v, s, t):
    '''⟨u, v⟩ is the common side;
    ⟨s, t⟩ are the opposing vertices;
    returns False also if it is a triangle
    only works if ⟨s, t⟩ crosses the line defined by ⟨u, v⟩'''
    # this used to be called `is_quadrilateral_convex()`
    # us × ut
    usut = np.cross(s - u, t - u)
    # vt × vs
    vtvs = np.cross(t - v, s - v)
    if usut == 0 or vtvs == 0:
        # the four vertices form a triangle
        return False
    return (usut > 0) == (vtvs > 0)


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


def is_blocking(root, u, v, w, y):
    # w and y are necessarily on opposite sides of uv
    # (because of Delaunay – see the triangles construction)
    # hence, if (root, y) are on the same side, (w, root) are not
    return (is_triangle_pair_a_convex_quadrilateral(u, v, w, root)
            if is_same_side(u, v, root, y)
            else is_triangle_pair_a_convex_quadrilateral(u, v, root, y))


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
    M = G.graph['M']
    # N = G.number_of_nodes() - M
    VertexC = G.graph['VertexC']
    # roots = range(N, N + M)
    roots = range(-M, 0)
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
        for n_inner in (N_inner & set([v for u, v in G.edges(n_hull)])):
            uv = frozenset((n_hull, n_inner))
            u, v = uv
            opposites = triangles[uv]
            if len(opposites) == 2:
                w, y = triangles[uv]
                rootC = VertexC[G.edges[u, v]['root']]
                uvwyC = tuple((VertexC[n] for n in (*uv, w, y)))
                if not is_blocking(rootC, *uvwyC):
                    E_hull_exp.add(uv)
                    G.edges[uv]['exempted'] = True

    # calculate blockage arc for each edge
    zeros = np.full((M,), 0.)
    for u, v, d in list(G.edges(data=True)):
        if (frozenset((u, v)) in E_hull_exp) or (u in roots) or (v in roots):
            angdiff = zeros
        else:
            # angdiff = (angles[:, u] - angles[:, v]) % (2*np.pi)
            # angdiff = abs(angles[:, u] - angles[:, v])
            angdiff = abs(angles[u] - angles[v])
        arc = np.empty((M,), dtype=float)
        for i in range(M):  # TODO: vectorize this loop
            arc[i] = angdiff[i] if angdiff[i] < np.pi else 2*np.pi - angdiff[i]
        d['arc'] = arc
        # if arc is π/2 or more, remove the edge (it's shorter to go to root)
        if allow_edge_deletion and any(arc >= np.pi/2):
            G.remove_edge(u, v)
            print('angles', arc, 'removing «',
                  '–'.join([F[n] for n in (u, v)]), '»')


def edge_crossings(s, t, G, diagonals, P):
    s, t = (s, t) if s < t else (t, s)
    v = diagonals.get((s, t))
    crossings = []
    if v is None:
        # ⟨s, t⟩ is a Delaunay edge
        Pst = P[s][t]
        Pts = P[t][s]
        u = Pst['cw']
        v = Pts['cw']
        if u == Pts['ccw'] and v == Pst['ccw']:
            diag = (u, v) if u < v else (v, u)
            if diag in diagonals and diag in G.edges:
                crossings.append(diag)
    else:
        # ⟨s, t⟩ is a diagonal
        u = P[v][s]['cw']
        triangles = ((u, v, s), (v, u, t))
        u, v = (u, v) if u < v else (v, u)
        # crossing with Delaunay edge
        crossings.append((u, v))
        # examine the two triangles (u, v) belongs to
        for a, b, c in triangles:
            # this is for diagonals crossing diagonals
            d = P[c][b]['cw']
            diag_da = (a, d) if a < d else (d, a)
            if d == P[b][c]['ccw'] and diag_da in diagonals:
                crossings.append(diag_da)
            e = P[a][c]['cw']
            diag_eb = (e, b) if e < b else (b, e)
            if e == P[c][a]['ccw'] and diag_eb in diagonals:
                crossings.append(diag_eb)
    return [edge for edge in crossings if edge in G.edges]


def make_planar_embedding(M: int, VertexC: np.ndarray,
                          max_tri_AR=MAX_TRIANGLE_ASPECT_RATIO):
    V = VertexC.shape[0]
    N = V - M
    SwappedVertexC = np.vstack((VertexC[-M:], VertexC[:N]))
    tri = Delaunay(SwappedVertexC)

    NULL = np.iinfo(tri.simplices.dtype).min
    mat = np.full((V, V), NULL, dtype=tri.simplices.dtype)

    S = tri.simplices - M
    mat[S[:, 0], S[:, 1]] = S[:, 2]
    mat[S[:, 1], S[:, 2]] = S[:, 0]
    mat[S[:, 2], S[:, 0]] = S[:, 1]
    del S

    # getting rid of nearly flat Delaunay triangles
    # qhull (used by scipy) seems not able to do it
    # reference: http://www.qhull.org/html/qh-faq.htm#flat
    old_hull = [(u, v) if u < v else (v, u) for u, v in (tri.convex_hull - M)]
    new_hull = []
    while old_hull:
        u, v = old_hull.pop()
        t = max(mat[u, v], mat[v, u])
        AR = triangle_AR(*VertexC[(u, v, t),])
        if AR > max_tri_AR:
            # TODO: document this relaxation of MAX_ASPECT for root nodes
            if min(u, v, t) < 0 and AR < 50*max_tri_AR:
                # when considering root nodes, be less strict with AR
                if (u, v) not in new_hull:
                    new_hull.append((u, v))
                continue
            # print(ut, vt, AR)
            ut = (u, t) if u < t else (t, u)
            if ut not in old_hull:
                old_hull.append(ut)
            vt = (v, t) if v < t else (t, v)
            if vt not in old_hull:
                old_hull.append(vt)
            mat[u, v] = mat[v, u] = NULL
            for a, b, c in ((u, t, v), (t, u, v), (v, t, u), (t, v, u)):
                if mat[a, b] == c:
                    mat[a, b] = NULL
        else:
            if (u, v) not in new_hull:
                new_hull.append((u, v))

    planar = nx.PlanarEmbedding()
    planar.add_nodes_from(range(-M, N))
    # add planar embedding half-edges, using
    # Delaunay triangles (vertices in ccw order)
    # triangles are stored in `mat`
    # i.e. mat[u, v] == t if u, v, t are vertices
    # of a triangle in ccw order
    # for u, next_ in enumerate(mat, start=-M):

    # diagonals store an edge ⟨s, t⟩ as key (s < t)
    # and a reference node `v` as value; to add a
    # diagonal to a PlanarEmbedding use these two lines:
    #     PlanarEmbedding.add_half_edge_ccw(s, t, v)
    #     PlanarEmbedding.add_half_edge_cw(t, s, v)
    # to find u, one can use:
    #     _, u = PlanarEmbedding.next_face_half_edge(v, s)
    # or:
    #     u = PlanarEmbedding[v][s]['cw']
    diagonals = {}

    for u, next_ in zip(chain(range(N), range(-M, 0)), mat):
        # first rotate ccw, so next_ is a row
        # get any of node's edge
        # argmax() of boolean may use shortcircuiting logic
        # which means it would stop searching on the first True
        first = (next_ >= 0).argmax()
        first = first % N - M*(first//N)
        v = first
        back = mat[v, u]
        fwd = next_[v]
        planar.add_half_edge_first(u, v)
        if back != NULL and fwd != NULL:
            uC, vC, fwdC, backC = VertexC[(u, v, fwd, back),]
            s, t = (back, fwd) if back < fwd else (fwd, back)
            if ((s, t) not in diagonals
                    and triangle_AR(fwdC, uC, backC) < max_tri_AR
                    and triangle_AR(fwdC, vC, backC) < max_tri_AR
                    and is_triangle_pair_a_convex_quadrilateral(uC, vC, backC,
                                                                fwdC)):
                diagonals[(s, t)] = v if s == back else u
        # start by circling vertex u in ccw direction
        add = planar.add_half_edge_ccw
        ccw = True
        # when fwd == first, all triangles around vertex u have been visited
        while fwd != first:
            if fwd != NULL:
                back = v
                v = fwd
                fwd = next_[v]
                add(u, v, back)
                if fwd != NULL:
                    uC, vC, fwdC, backC = VertexC[(u, v, fwd, back),]
                    s, t = (back, fwd) if back < fwd else (fwd, back)
                    if ((s, t) not in diagonals
                            and triangle_AR(fwdC, uC, backC) < max_tri_AR
                            and triangle_AR(fwdC, vC, backC) < max_tri_AR
                            and is_triangle_pair_a_convex_quadrilateral(
                                uC, vC, backC, fwdC)):
                        if ccw:
                            diagonals[(s, t)] = v if s == back else u
                        else:
                            diagonals[(s, t)] = u if s == back else v
            elif ccw:
                # ccw direction reached the convex hull
                # start from first again in cw direction
                add = planar.add_half_edge_cw
                ccw = False
                # when going cw, next_ is a column
                next_ = mat[:, u]
                back = mat[u, first]
                v = first
                fwd = next_[v]
            else:
                # cw direction ended at the convex hull
                break
    del mat
    # raise an exception if `planar` is not proper:
    planar.check_structure()
    return planar, diagonals


def delaunay(G_base, add_diagonals=True, debug=False, bind2root=False,
             max_tri_AR=MAX_TRIANGLE_ASPECT_RATIO, **qhull_options):
    '''Creates a networkx graph from the Delaunay triangulation
    of the point in coordinates. Each edge gets an attribute `length`
    with the euclidean distance between its vertices.'''
    M = G_base.graph['M']
    VertexC = G_base.graph['VertexC']
    N = VertexC.shape[0] - M

    planar, diagonals = make_planar_embedding(
            M, VertexC, max_tri_AR=MAX_TRIANGLE_ASPECT_RATIO)

    # undirected Delaunay edge view
    undirected = planar.to_undirected(as_view=True)

    # build the undirected graph
    A = nx.Graph()
    A.add_nodes_from(((n, {'label': label})
                      for n, label in G_base.nodes(data='label')
                      if 0 <= n < N), type='wtg')
    for r in range(-M, 0):
        A.add_node(r, label=G_base.nodes[r]['label'], type='oss')
    A.add_edges_from(undirected.edges, type='delaunay')
    E_planar = np.array(undirected.edges, dtype=int)
    Length = np.hypot(*(VertexC[E_planar[:, 0]] - VertexC[E_planar[:, 1]]).T)
    for (u, v), length in zip(E_planar, Length):
        A[u][v]['length'] = length
    if add_diagonals:
        diagnodes = np.empty((len(diagonals), 2), dtype=int)
        for row, uv in zip(diagnodes, diagonals):
            row[:] = uv
        A.add_edges_from(diagonals, type='extended')
        # the reference vertex `v` that `diagonals` carries
        # could be stored as edge ⟨s, t⟩'s property (that
        # property would also mark the edge as a diagonal)
        Length = np.hypot(*(VertexC[diagnodes[:, 0]]
                            - VertexC[diagnodes[:, 1]]).T)
        for (u, v), length in zip(diagnodes, Length):
            A[u][v]['length'] = length

    d2roots = G_base.graph.get('d2roots')
    if d2roots is None:
        d2roots = cdist(VertexC[:-M], VertexC[-M:])
    if bind2root:
        for n, n_root in G_base.nodes(data='root'):
            A.nodes[n]['root'] = n_root
        # alternatively, if G_base nodes do not have 'root' attr:
        #  for n, nodeD in A.nodes(data=True):
        #      nodeD['root'] = -M + np.argmin(d2roots[n])
        # assign each edge to the root closest to the edge's middle point
        for u, v, edgeD in A.edges(data=True):
            edgeD['root'] = -M + np.argmin(
                    cdist(((VertexC[u] + VertexC[v])/2)[np.newaxis, :],
                          VertexC[-M:]))
    A.graph.update(M=M,
                   VertexC=VertexC,
                   planar=planar,
                   d2roots=d2roots,
                   diagonals=diagonals,
                   landscape_angle=G_base.graph.get('landscape_angle', 0),
                   boundary=G_base.graph['boundary'],
                   name=G_base.graph['name'],
                   handle=G_base.graph['handle'])

    # TODO: update other code that uses the data below
    # old version of delaunay() also stored these:
    # save the convex hull node set
    # G.graph['N_hull'] = N_hull  # only in geometric.py
    # save the convex hull edge set
    # G.graph['E_hull'] = E_hull  # only in geometric.py

    # these two are more important, as they are used in EW
    # variants that do crossing checks (calls to `edge_crossings()`)
    # G.graph['triangles'] = triangles
    # G.graph['triangles_exp'] = triangles_exp
    return A


def make_graph_metrics(G):
    '''
    This function changes G in place!
    Calculates for all nodes, for each root node:
    - distance to root nodes
    - angle wrt root node

    Any detour nodes in G are ignored.
    '''
    VertexC = G.graph['VertexC']
    M = G.graph['M']
    # N = G.number_of_nodes() - M
    roots = range(-M, 0)
    NodeC = VertexC[:-M]
    RootC = VertexC[-M:]

    # calculate distance from all nodes to each of the roots
    d2roots = cdist(VertexC[:-M], VertexC[-M:])

    angles = np.empty_like(d2roots)
    for n, nodeC in enumerate(NodeC):
        nodeD = G.nodes[n]
        # assign the node to the closest root
        nodeD['root'] = -M + np.argmin(d2roots[n])
        x, y = (nodeC - RootC).T
        angles[n] = np.arctan2(y, x)
    # TODO: ¿is this below actually used anywhere?
    # assign root nodes to themselves (for completeness?)
    for root in roots:
        G.nodes[root]['root'] = root

    G.graph['d2roots'] = d2roots
    G.graph['d2rootsRank'] = np.argsort(np.argsort(d2roots, axis=0), axis=0)
    G.graph['angles'] = angles
    G.graph['anglesRank'] = np.argsort(np.argsort(angles, axis=0), axis=0)
    G.graph['anglesYhp'] = angles >= 0.
    G.graph['anglesXhp'] = abs(angles) < np.pi/2


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
def complete_graph(G_base, include_roots=False, prune=True, crossings=False):
    '''Creates a networkx graph connecting all non-root nodes to every
    other non-root node. Edges with an arc > pi/2 around root are discarded
    The length of each edge is the euclidean distance between its vertices.'''
    M = G_base.graph['M']
    VertexC = G_base.graph['VertexC']
    N = VertexC.shape[0] - M
    NodeC = VertexC[:-M]
    RootC = VertexC[-M:]
    Root = range(-M, 0)
    V = N + (M if include_roots else 0)
    G = nx.complete_graph(V)
    EdgeComplete = np.column_stack(np.triu_indices(V, k=1))
    #  mask = np.zeros((V,), dtype=bool)
    mask = np.zeros_like(EdgeComplete[0], dtype=bool)
    if include_roots:
        # mask root-root edges
        offset = 0
        for i in range(0, M - 1):
            for j in range(0, M - i - 1):
                mask[offset + j] = True
            offset += (V - i - 1)

        # make node indices span -M:(N - 1)
        EdgeComplete -= M
        nx.relabel_nodes(G, dict(zip(range(N, N + M), Root)),
                         copy=False)
        C = cdist(VertexC, VertexC)
    else:
        C = cdist(NodeC, NodeC)
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
    if crossings:
        # get_crossings_map() takes time and space
        G.graph['crossings'] = get_crossings_map(Edge, VertexC)
    # assign nodes to roots?
    # remove edges between nodes belonging to distinct roots whose length is
    # greater than both d2root
    G.graph.update(G_base.graph)
    nx.set_node_attributes(G, G_base.nodes)
    for u, v, edgeD in G.edges(data=True):
        edgeD['length'] = C[u, v]
        # assign the edge to the root closest to the edge's middle point
        edgeD['root'] = -M + np.argmin(
            cdist(((VertexC[u] + VertexC[v])/2)[np.newaxis, :], RootC))
    return G


def A_graph(G_base, delaunay_based=True, weightfun=None, weight_attr='weight'):
    '''Return the "available edges" graph that is the base for edge search in
    Esau-Williams. If `delaunay_based` is True, the edges are the expanded
    Delaunay triangulation, otherwise a complete graph is returned.'''
    if delaunay_based:
        A = delaunay(G_base)
        if weightfun is not None:
            apply_edge_exemptions(A)
    else:
        A = complete_graph(G_base, include_roots=True)
        # intersections
        # I = get_crossings_list(np.array(A.edges()), VertexC)

    if weightfun is not None:
        for u, v, data in A.edges(data=True):
            data[weight_attr] = weightfun(data)

    # remove all gates from A
    # TODO: decide about this line
    # A.remove_edges_from(list(A.edges(range(-M, 0))))
    return A


def planar_over_layout(G: nx.Graph):
    '''
    Return a PlanarEmbedding of a triangulation of the nodes in G, provided
    G has been created using the extended Delaunay edges.

    The PlanarEmbedding is different from the one generated in `A_graph()` in
    that it takes into account the actual edges used in G (i.e. used diagonals
    will be included in the planar graph to the exclusion of Delaunay edges
    that cross them).
    '''
    M = G.graph['M']
    VertexC = G.graph['VertexC']
    P, diagonals = make_planar_embedding(M, VertexC)
    for r in range(-M, 0):
        for u, v in nx.edge_dfs(G, r):
            # update the planar embedding to include any Delaunay diagonals
            # used in G; the corresponding crossing Delaunay edge is removed
            u, v = (u, v) if u < v else (v, u)
            s = diagonals.get((u, v))
            if s is not None:
                t = P[u][s]['ccw']  # same as P[v][s]['cw']
                if (s, t) in G.edges:
                    # (u, v) & (s, t) are in G (i.e. a crossing). This means
                    # the diagonal (u, v) is a gate and (s, t) should remain
                    continue
                P.add_half_edge_cw(u, v, t)
                P.add_half_edge_cw(v, u, s)
                P.remove_edge(s, t)
    P.graph['diagonals'] = diagonals
    return P


def minimum_spanning_tree(G: nx.Graph) -> nx.Graph:
    '''Return a graph of the minimum spanning tree connecting the node in G.'''
    M = G.graph['M']
    VertexC = G.graph['VertexC']
    V = VertexC.shape[0]
    N = V - M
    P = make_planar_embedding(M, VertexC)[0].to_undirected(as_view=True)
    E_planar = np.array(P.edges, dtype=np.int32)
    # E_planar = np.array(P.edges)
    Length = np.hypot(*(VertexC[E_planar[:, 0]] - VertexC[E_planar[:, 1]]).T)
    E_planar[E_planar < 0] += V
    P_ = coo_array((Length, (*E_planar.T,)), shape=(V, V))
    Q_ = scipy_mst(P_)
    S, T = Q_.nonzero()
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for s, t in zip(S, T):
        H.add_edge(s if s < N else s - V, t if t < N else t - V,
                   length=Q_[s, t])
    H.graph.update(G.graph)
    return H


# TODO: MARGIN is ARBITRARY - depends on the scale
def check_crossings(G, debug=False, MARGIN=0.1):
    '''Checks for crossings (touch/overlap is not considered crossing).
    This is an independent check on the tree resulting from the heuristic.
    It is not supposed to be used within the heuristic.
    MARGIN is how far an edge can advance across another one and still not be
    considered a crossing.'''
    VertexC = G.graph['VertexC']
    M = G.graph['M']
    N = G.number_of_nodes() - M

    D = G.graph.get('D')
    if D is not None:
        N -= D
        # detournodes = range(N, N + D)
        # G.add_nodes_from(((s, {'type': 'detour'})
        #                   for s in detournodes))
        # clone2prime = G.graph['clone2prime']
        # assert len(clone2prime) == D, \
        #     'len(clone2prime) != D'
        # fnT = np.arange(N + D + M)
        # fnT[N: N + D] = clone2prime
        # DetourC = VertexC[clone2prime].copy()
        fnT = G.graph['fnT']
        AllnodesC = np.vstack((VertexC[:N], VertexC[fnT[N:N + D]],
                               VertexC[-M:]))
    else:
        fnT = np.arange(N + M)
        AllnodesC = VertexC
    roots = range(-M, 0)
    fnT[-M:] = roots
    n2s = NodeStr(fnT, N)

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

    for root in roots:
        # edges = list(nx.edge_dfs(G, source=root))
        edges = list(nx.edge_bfs(G, source=root))
        # outstr = ', '.join([f'«{F[fnT[u]]}–{F[fnT[v]]}»' for u, v in edges])
        # print(outstr)
        potential = []
        for i, (u, v) in enumerate(edges):
            for s, t in edges[(i + 1):]:
                if s == u or s == v:
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
                                        print(f'ERROR <partial ovelap>: edge '
                                              f'«{F[fnT[u]]}–{F[fnT[v]]}» '
                                              f'crosses '
                                              f'«{F[fnT[s]]}–{F[fnT[t]]}»')
                                        crossings.append(((u,  v), (s, t)))
    debug and potential and print(
        'potential crossings: ' +
        ', '.join([f'«{F[fnT[u]]}–{F[fnT[v]]}» × «{F[fnT[s]]}–{F[fnT[t]]}»'
                   for u, v, s, t in potential]))
    return crossings
