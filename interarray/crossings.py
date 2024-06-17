import operator
import math
import numpy as np
from .geometric import is_same_side, make_graph_metrics
import networkx as nx


def get_crossings_list(Edge, VertexC):
    '''
    List all crossings between edges in the `Edge` (E×2) numpy array.
    Coordinates must be provided in the `VertexC` (V×2) array.

    Used when edges are not limited to the expanded Delaunay set.
    '''
    crossings = []
    V = VertexC[Edge[:, 1]] - VertexC[Edge[:, 0]]
    for i, ((UVx, UVy), (u, v)) in enumerate(zip(V, Edge[:-1])):
        uCx, uCy = VertexC[u]
        vCx, vCy = VertexC[v]
        for j, ((STx, STy), (s, t)) in enumerate(zip(-V[i+1:], Edge[i+1:],
                                                     start=i+1)):
            if s == u or t == u or s == v or t == v:
                # <edges have a common node>
                continue
            # bounding box check
            sCx, sCy = VertexC[s]
            tCx, tCy = VertexC[t]

            # X
            lo, hi = (vCx, uCx) if UVx < 0 else (uCx, vCx)
            if STx > 0:
                if hi < tCx or sCx < lo:
                    continue
            else:
                if hi < sCx or tCx < lo:
                    continue

            # Y
            lo, hi = (vCy, uCy) if UVy < 0 else (uCy, vCy)
            if STy > 0:
                if hi < tCy or sCy < lo:
                    continue
            else:
                if hi < sCy or tCy < lo:
                    continue

            # TODO: save the edges that have interfering bounding boxes
            #       to be checked in a vectorized implementation of
            #       the math below
            UV = UVx, UVy
            ST = STx, STy

            # denominator
            f = STx*UVy - STy*UVx
            # print('how close: ', f)
            # TODO: arbitrary threshold
            if math.isclose(f, 0, abs_tol=1e-3):
                # segments are parallel
                continue

            C = uCx - sCx, uCy - sCy
            # alpha and beta numerators
            for num in (Px*Qy - Py*Qx for (Px, Py), (Qx, Qy) in ((C, ST),
                                                                 (UV, C))):
                if f > 0:
                    if less(num, 0) or less(f, num):
                        continue
                else:
                    if less(0, num) or less(num, f):
                        continue

            # segments do cross
            crossings.append((u, v, s, t))
    return crossings


def edgeXing_iter(u, v, G, A):
    '''This is broken, do not use! Use `edgeset_edgeXing_iter()` instead.'''
    planar = A.graph['planar']
    _, s = A.next_face_half_edge(u, v)
    _, t = A.next_face_half_edge(v, u)
    if s == t:
        # <u, v> and the 3rd vertex are hull
        return
    if (s, t) in A.edges:
        # the diagonal conflicts with the Delaunay edge
        yield ((u, v), (s, t))
        conflicting = [(s, t)]
    else:
        conflicting = []
    # examine the two triangles (u, v) belongs to
    for a, b, c in ((u, v, s),
                    (v, u, t)):
        # this is for diagonals crossing diagonals
        triangle = tuple(sorted((a, b, c)))
        if triangle not in checked:
            checked.add(triangle)
            _, e = A.next_face_half_edge(c, b)
            if (a, e) in A.edges:
                conflicting.append((a, e))
            _, d = A.next_face_half_edge(a, c)
            if (b, d) in A.edges:
                conflicting.append((b, d))
            if len(conflicting) > 1:
                yield conflicting


def layout_edgeXing_iter(G, A):
    '''does this even make sense?'''
    for edge in G.edges:
        yield from edgeXing_iter(edge, G, A)


def edgeset_edgeXing_iter(A):
    '''Iterator over all edge crossings in an expanded
    Delaunay edge set `A`. Each crossing is a 2 or 3-tuple
    of (u, v) edges.'''
    P = A.graph['planar']
    diagonals = A.graph['diagonals']
    checked = set()
    for (s, t), v in diagonals.items():
        u = P[v][s]['cw']
        triangles = ((u, v, s), (v, u, t))
        u, v = (u, v) if u < v else (v, u)
        # crossing with Delaunay edge
        yield ((u, v), (s, t))
        # examine the two triangles (u, v) belongs to
        for a, b, c in triangles:
            triangle = tuple(sorted((a, b, c)))
            if triangle in checked:
                continue
            checked.add(triangle)
            # this is for diagonals crossing diagonals
            conflicting = [(s, t)]
            d = P[c][b]['cw']
            diag_da = (a, d) if a < d else (d, a)
            if d == P[b][c]['ccw'] and diag_da in diagonals:
                conflicting.append(diag_da)
            e = P[a][c]['cw']
            diag_eb = (e, b) if e < b else (b, e)
            if e == P[c][a]['ccw'] and diag_eb in diagonals:
                conflicting.append(diag_eb)
            if len(conflicting) > 1:
                yield conflicting


def edgeset_edgeXing_iter_deprecated(A, include_roots=False):
    '''DEPRECATED!

    Iterator over all edge crossings in an expanded
    Delaunay edge set `A`. Each crossing is a 2 or 3-tuple
    of (u, v) edges.'''
    planar = A.graph['planar']
    checked = set()
    # iterate over all Delaunay edges
    for u, v in planar.edges:
        if u > v or (not include_roots and (u < 0 or v < 0)):
            # planar is a DiGraph, so skip one half-edge of the pair
            continue
        # get diagonal
        _, s = planar.next_face_half_edge(u, v)
        _, t = planar.next_face_half_edge(v, u)
        if s == t or (not include_roots and (s < 0 or t < 0)):
            # <u, v> and the 3rd vertex are hull
            continue
        triangles = []
        if (s, u) in planar.edges:
            triangles.append((u, v, s))
        if (t, v) in planar.edges:
            triangles.append((v, u, t))
        s, t = (s, t) if s < t else (t, s)
        has_diagonal = (s, t) in A.edges
        if has_diagonal:
            # the diagonal conflicts with the Delaunay edge
            yield ((u, v), (s, t))
        # examine the two triangles (u, v) belongs to
        for a, b, c in triangles:
            # this is for diagonals crossing diagonals
            triangle = tuple(sorted((a, b, c)))
            if triangle not in checked:
                checked.add(triangle)
                conflicting = [(s, t)] if has_diagonal else []
                _, e = planar.next_face_half_edge(c, b)
                if ((e, c) in planar.edges
                        and (a, e) in A.edges
                        and (include_roots or (a >= 0 and e >= 0))):
                    conflicting.append((a, e) if a < e else (e, a))
                _, d = planar.next_face_half_edge(a, c)
                if ((d, a) in planar.edges
                        and (b, d) in A.edges
                        and (include_roots or (b >= 0 and d >= 0))):
                    conflicting.append((b, d) if b < d else (d, b))
                if len(conflicting) > 1:
                    yield conflicting


# adapted edge_crossings() from geometric.py
# delaunay() does not create `triangles` and `triangles_exp`
# anymore, so this is broken
def edgeXing_iter_deprecated(A):
    '''
    DEPRECATED!
    This is broken, do not use!

    Iterates over all pairs of crossing edges in `A`. This assumes `A`
    has only expanded Delaunay edges (with triangles and triangles_exp).

    Used in constraint generation for MILP model.
    '''
    triangles = A.graph['triangles']
    # triangles_exp maps expanded Delaunay to Delaunay edges
    triangles_exp = A.graph['triangles_exp']
    checked = set()
    for uv, (s, t) in triangles_exp.items():
        # <(u, v) is an expanded Delaunay edge>
        u, v = uv
        checked.add(uv)
        if (u, v) not in A.edges:
            continue
        if (s, t) in A.edges:
            yield (((u, v) if u < v else (v, u)),
                   ((s, t) if s < t else (t, s)))
        else:
            # this looks wrong...
            # the only case where this might happen is
            # when a Delaunay edge is removed because of
            # the angle > pi/2 blocking of a root node
            # but even in this case, we should check for
            # crossings with other expanded edges
            continue
        for a_b in ((u, s), (u, t), (s, v), (t, v)):
            if a_b not in triangles:
                continue
            cd = triangles[frozenset(a_b)]
            if cd in checked:
                continue
            if (cd in triangles_exp
                    and tuple(cd) in A.edges
                    # this last condition is for edges that should have been
                    # eliminated in delaunay()'s hull_edge_is_overlapping(),
                    # but weren't
                    and set(triangles_exp[cd]) <= {u, v, s, t}):
                c, d = cd
                yield (((u, v) if u < v else (v, u)),
                       ((c, d) if c < d else (d, c)))


def gateXing_iter(G, gates=None, touch_is_cross=True):
    '''
    Iterate over all crossings between non-gate edges and the edges in `gates`.
    If `gates` is None, all nodes that are not a root neighbor are considered.
    Arguments:
    - `gates`: sequence of #root sequences of gate nodes; if None, all nodes
    - `touch_is_cross`: if True, count as crossing a gate going over a node

    The order of items in `gates` must correspond to roots in range(-M, 0).
    Used in constraint generation for MILP model.
    '''
    M = G.graph['M']
    VertexC = G.graph['VertexC']
    N = VertexC.shape[0] - M
    roots = range(-M, 0)
    anglesRank = G.graph.get('anglesRank', None)
    if anglesRank is None:
        make_graph_metrics(G)
        anglesRank = G.graph['anglesRank']
    anglesXhp = G.graph['anglesXhp']
    anglesYhp = G.graph['anglesYhp']
    # iterable of non-gate edges:
    Edge = nx.subgraph_view(G, filter_node=lambda n: n >= 0).edges()
    if gates is None:
        all_nodes = set(range(N))
        IGate = []
        for r in roots:
            nodes = all_nodes.difference(G.neighbors(r))
            IGate.append(np.fromiter(nodes, dtype=int, count=len(nodes)))
    else:
        IGate = gates
    # it is important to consider touch as crossing
    # because if a gate goes precisely through a node
    # there will be nothing to prevent it from spliting
    # that node's subtree
    less = operator.le if touch_is_cross else operator.lt
    for u, v in Edge:
        uC = VertexC[u]
        vC = VertexC[v]
        for root, iGate in zip(roots, IGate):
            rootC = VertexC[root]
            uR, vR = anglesRank[u, root], anglesRank[v, root]
            highRank, lowRank = (uR, vR) if uR >= vR else (vR, uR)
            Xhp = anglesXhp[[u, v], root]
            uYhp, vYhp = anglesYhp[[u, v], root]
            # get a vector of gate edges' ranks for current root
            gaterank = anglesRank[iGate, root]
            # check if angle of <u, v> wraps across +-pi
            if (not any(Xhp)) and uYhp != vYhp:
                # <u, v> wraps across zero
                is_rank_within = np.logical_or(less(gaterank, lowRank),
                                               less(highRank, gaterank))
            else:
                # <u, v> does not wrap across zero
                is_rank_within = np.logical_and(less(lowRank, gaterank),
                                                less(gaterank, highRank))
            for n in iGate[np.flatnonzero(is_rank_within)]:
                # this test confirms the crossing because `is_rank_within`
                # established that root–n is on a line crossing u–v
                if not is_same_side(uC, vC, rootC, VertexC[n]):
                    u, v = (u, v) if u < v else (v, u)
                    yield (u, v), (root, n)
