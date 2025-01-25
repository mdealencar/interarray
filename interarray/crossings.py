import operator
import math
from collections.abc import Iterator, Iterable
from bidict import bidict
from itertools import chain
import numpy as np
import networkx as nx
from .interarraylib import calcload
from .geometric import (
    is_same_side,
    is_bunch_split_by_corner,
    angle_helpers
)


def get_interferences_list(Edge: np.ndarray, VertexC: np.ndarray,
                           fnT: np.ndarray | None = None,
                           EPSILON=1e-15) -> list:
    '''
    List all crossings between edges in the `Edge` (E×2) numpy array.
    Coordinates must be provided in the `VertexC` (V×2) array.

    `Edge` contains indices to VertexC. If `Edge` includes detour nodes
    (i.e. indices go beyond `VertexC`'s length), `fnT` translation table
    must be provided.

    Should be used when edges are not limited to the expanded Delaunay set.
    '''
    crossings = []
    if fnT is None:
        V = VertexC[Edge[:, 1]] - VertexC[Edge[:, 0]]
    else:
        V = VertexC[fnT[Edge[:, 1]]] - VertexC[fnT[Edge[:, 0]]]
    for i, ((UVx, UVy), (u, v)) in enumerate(zip(V[:-1], Edge[:-1])):
        u_, v_ = (u, v) if fnT is None else fnT[[u, v]]
        (uCx, uCy), (vCx, vCy) = VertexC[[u_, v_]]
        for j, ((STx, STy), (s, t)) in enumerate(zip(-V[i+1:], Edge[i+1:]),
                                                 start=i+1):
            s_, t_ = (s, t) if fnT is None else fnT[[s, t]]
            if s_ == u_ or t_ == u_ or s_ == v_ or t_ == v_:
                # <edges have a common node>
                continue
            # bounding box check
            (sCx, sCy), (tCx, tCy) = VertexC[[s_, t_]]

            # X
            lo, hi = (vCx, uCx) if UVx < 0 else (uCx, vCx)
            if STx > 0:  # s - t > 0 -> hi: s, lo: t
                if hi < tCx or sCx < lo:
                    continue
            else:  # s - t < 0 -> hi: t, lo: s
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
            # TODO: verify if this arbitrary tolerance is appropriate
            if math.isclose(f, 0., abs_tol=1e-5):
                # segments are parallel
                # TODO: there should be check for branch splitting in parallel
                #       cases with touching points
                continue

            C = uCx - sCx, uCy - sCy
            touch_found = []
            Xcount = 0
            for k, num in enumerate((Px*Qy - Py*Qx)
                                    for (Px, Py), (Qx, Qy) in
                                    ((C, ST), (UV, C))):
                if f > 0:
                    if -EPSILON <= num <= f + EPSILON:  # num < 0 or f < num:
                        Xcount += 1
                        if math.isclose(num, 0, abs_tol=EPSILON):
                            touch_found.append(2*k)
                        if math.isclose(num, f, abs_tol=EPSILON):
                            touch_found.append(2*k + 1)
                else:
                    if f - EPSILON <= num <= EPSILON:  # 0 < num or num < f:
                        Xcount += 1
                        if math.isclose(num, 0, abs_tol=EPSILON):
                            touch_found.append(2*k)
                        if math.isclose(num, f, abs_tol=EPSILON):
                            touch_found.append(2*k + 1)

            if Xcount == 2:
                # segments cross or touch
                uvst = (u, v, s, t)
                if touch_found:
                    assert len(touch_found) == 1, \
                        'ERROR: too many touching points.'
                    #  p = uvst[touch_found[0]]
                    p = touch_found[0]
                else:
                    p = None
                crossings.append((uvst, p))
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


def edge_crossings(u: int, v: int, G: nx.Graph, diagonals: bidict) \
        -> list[tuple[int, int]]:
    u, v = (u, v) if u < v else (v, u)
    st = diagonals.get((u, v))
    conflicting = []
    if st is None:
        # ⟨u, v⟩ is a Delaunay edge
        st = diagonals.inv.get((u, v))
        if st is not None and st[0] >= 0:
            conflicting.append(st)
    else:
        # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
        s, t = st
        # crossing with Delaunay edge
        conflicting.append(st)

        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in (u, v):
            for diag in (diagonals.inv.get((w, y) if w < y else (y, w))
                         for w, y in ((s, hat), (hat, t))):
                if diag is not None and diag[0] >= 0:
                    conflicting.append(diag)
    return [edge for edge in conflicting if edge in G.edges]


def edgeset_edgeXing_iter(diagonals: bidict) \
        -> Iterator[list[tuple[int, int]]]:
    '''
    Iterator over all edge crossings in an expanded Delaunay edge set `A`.
    Each crossing is a 2 or 3-tuple of (u, v) edges. Does not include gates.
    '''
    checked = set()
    for (u, v), (s, t) in diagonals.items():
        # ⟨u, v⟩ is a diagonal of Delaunay ⟨s, t⟩
        if u < 0:
            # diagonal is a gate
            continue
        uv = (u, v)
        if s >= 0:
            # crossing with Delaunay edge
            yield ((s, t), uv)
        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in uv:
            triangle = tuple(sorted((s, t, hat)))
            if triangle in checked:
                continue
            checked.add(triangle)
            conflicting = [uv]
            for diag in (diagonals.inv.get((w, y) if w < y else (y, w))
                         for w, y in ((s, hat), (hat, t))):
                if diag is not None and diag[0] >= 0:
                    conflicting.append(diag)
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


def gateXing_iter(G: nx.Graph, *, hooks: Iterable | None = None,
                  borders: Iterable | None = None,
                  touch_is_cross: bool = True) \
        -> Iterator[tuple[tuple[int, int]]]:
    '''Iterate over all crossings between gates and edges/borders in G.

    If `hooks` is `None`, all nodes that are not a root neighbor are
    considered. Used in constraint generation for ILP model.

    Args:
        G: Routeset or edgeset (A) to examine.
        hooks: Nodes to check, grouped by root in sub-sequences from root `-R`
            to `-1`. If `None`, all non-root nodes are checked using `'root'`
            node attribute.
        borders: Impassable line segments between border vertices.
        touch_is_cross: If `True`, count as crossing a gate going over a node.

    Yields:
        Pair of (edge, gate) that cross (each a 2-tuple of nodes).
    '''
    R, T, VertexC = (G.graph[k] for k in ('R', 'T', 'VertexC'))
    fnT = G.graph.get('fnT')
    roots = range(-R, 0)
    anglesRank = G.graph.get('anglesRank', None)
    if anglesRank is None:
        _, anglesRank, anglesXhp, anglesYhp = angle_helpers(G)
    else:
        anglesXhp = G.graph['anglesXhp']
        anglesYhp = G.graph['anglesYhp']
    # TODO: There is a corner case here: for multiple roots, the gates are not
    #       being checked between different roots. Unlikely but possible case.
    # iterable of non-gate edges:
    Edge = nx.subgraph_view(G, filter_node=lambda n: n >= 0).edges()
    if borders is not None:
        Edge = chain(Edge, borders)
    if hooks is None:
        all_nodes = np.arange(T)
        IGate = [all_nodes]*R
    else:
        IGate = hooks
    # it is important to consider touch as crossing
    # because if a gate goes precisely through a node
    # there will be nothing to prevent it from spliting
    # that node's subtree
    less = operator.le if touch_is_cross else operator.lt
    for u, v in Edge:
        if fnT is not None:
            u, v = fnT[u], fnT[v]
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
            for n in iGate[np.flatnonzero(is_rank_within)].tolist():
                # this test confirms the crossing because `is_rank_within`
                # established that root–n is on a line crossing u–v
                if n == u or n == v:
                    continue
                if not is_same_side(uC, vC, rootC, VertexC[n]):
                    u, v = (u, v) if u < v else (v, u)
                    yield (u, v), (root, n)


def validate_routeset(G: nx.Graph) -> list[tuple[int, int, int, int]]:
    '''
    Check if route set represented by G's edges is topologically sound,
    repects capacity and has no edge crossings nor branch splitting.

    Returns:
        list of crossings/splits

    Example:
        F = NodeTagger()
        Xings = validate_routeset(G)
            for u, v, s, t in Xings:
                if u != v:
                    print(f'{F[u]}–{F[v]} crosses {F[s]}–{F[t]}')
                else:
                    print(f'detour @ {F[u]} splits {F[s]}–{F[v]}–{F[t]}')
    '''
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC']
    if C > 0 or D > 0:
        fnT = G.graph['fnT']
    else:
        fnT = np.arange(T + R)
        fnT[-R:] = range(-R, 0)

    # TOPOLOGY check: is it a proper tree?
    calcload(G)

    # TOPOLOGY check: is load within capacity?
    max_load = G.graph['max_load']
    capacity = G.graph.get('capacity')
    if capacity is not None:
        assert max_load <= capacity, f'κ = {capacity}, max_load= {max_load}'
    else:
        capacity = G.graph['capacity'] = max_load

    # check edge×edge crossings
    #  Edge = np.array(tuple((fnT[u], fnT[v]) for u, v in G.edges))
    XTings = get_interferences_list(np.array(G.edges), VertexC, fnT)
    # parallel is considered no crossing
    # analyse cases of touch
    Xings = []
    for uvst, p in XTings:
        if p is None:
            Xings.append(uvst)
            continue
        if G.degree[p] == 1:
            # trivial case: no way to break a branch apart
            continue
        # make u be the touch-point within ⟨s, t⟩
        if p < 2:
            u, v = uvst[:2] if p == 0 else uvst[1::-1]
            s, t = uvst[2:]
        else:
            u, v = uvst[2:] if p == 2 else uvst[:1:-1]
            s, t = uvst[:2]

        u_, v_, s_, t_ = fnT[uvst,]
        bunch = [fnT[nb] for nb in G[u]]
        is_split, insideI, outsideI = is_bunch_split_by_corner(
            VertexC[bunch], *VertexC[[s_, u_, t_]]
        )
        if is_split:
            Xings.append((s_, t_, bunch[insideI[0]], bunch[outsideI[0]]))

    # ¿do we need a special case for a detour segment going through a node?

    # check detour nodes for branch-splitting
    for d, d_ in zip(range(T, T + D), fnT[T:T + D]):
        if G.degree[d_] == 1:
            # trivial case: no way to break a branch apart
            continue
        dA, dB = (fnT[nb] for nb in G[d])
        bunch = [fnT[nb] for nb in G[d_]]
        is_split, insideI, outsideI = is_bunch_split_by_corner(
            VertexC[bunch], *VertexC[[dA, d_, dB]]
        )
        if is_split:
            Xings.append((d_, d_, bunch[insideI[0]], bunch[outsideI[0]]))
        # assert not is_split, \
        #     f'Detour around node {F[d_]} splits a branch; ' \
        #     f'inside: {[F[bunch[i]] for i in insideI]}; ' \
        #     f'outside: {[F[bunch[i]] for i in outsideI]}'
    return Xings


def list_edge_crossings(S: nx.Graph, A: nx.Graph) \
        -> list[tuple[tuple[int, int], tuple[int, int]]]:
    '''
    List edge×edge crossings for the network topology in S.
    `S` must only use extended Delaunay edges. It will not detect crossings
    of non-extDelaunay gates or detours.

    Args:
        S: solution topology
        A: available edges used in creating `S`

    Returns:
        list of 2-tuple (crossing) of 2-tuple (edge, ordered)
    '''
    eeXings = []
    checked = set()
    diagonals = A.graph['diagonals']
    for u, v in S.edges:
        u, v = (u, v) if u < v else (v, u)
        st = diagonals.get((u, v))
        if st is not None:
            # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
            if st in S.edges:
                # crossing with Delaunay edge ⟨s, t⟩
                eeXings.append((st, (u, v)))
            s, t = st
            # ⟨s, t⟩ may be part of up to two triangles, check their 4 sides
            sides = (((w, y) if w < y else (y, w))
                     for w, y in ((u, s), (s, v), (v, t), (t, u)))
            for side in sides:
                diag = diagonals.inv.get(side, False)
                if diag and diag in S.edges and diag not in checked:
                    checked.add((u, v))
                    eeXings.append((diag, (u, v)))
    return eeXings
