# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

from collections import defaultdict
from itertools import chain, tee

from gon import base as gonb

import PythonCDT as cdt

from interarray.geometric import (
    triangle_AR,
    is_triangle_pair_a_convex_quadrilateral,
    is_same_side,
    rotation_checkers_factory,
)
from interarray import MAX_TRIANGLE_ASPECT_RATIO
from interarray.interarraylib import NodeTagger

F = NodeTagger()


def edges_and_hull_from_cdt(triangles: list[cdt.Triangle],
                            vertmap: np.ndarray) -> list[tuple[int, int]]:
    '''
    THIS FUNCTION MAY BE IRRELEVANT, AS WE TYPICALLY NEED THE
    NetworkX.PlanarEmbedding ANYWAY, SO IT IS BETTER TO USE
    `planar_from_cdt_triangles()` DIRECTLY, followed by `hull_processor()`.

    Produces all the edges and a the convex hull (nodes) from a constrained
    Delaunay triangulation (via PythonCDT).

    `triangles` is a `PythonCDT.Triangulation().triangles` list
    `vertmap` is a node number translation table, from CDT numbers to NetworkX

    Returns:
    - list of edges that are sides of the triangles
    - list of nodes of the convex hull (counter-clockwise)
    '''
    tri_visited = set()
    hull_edges = {}

    def edges_from_tri(edge, tri_idx):
        # recursive function
        tri_visited.add(tri_idx)
        tri = triangles[tri_idx]
        b = (set(tri.vertices) - edge).pop()
        idx_b = tri.vertices.index(b)
        idx_c = (idx_b + 1) % 3
        idx_a = (idx_b - 1) % 3
        a = tri.vertices[idx_a]
        AB = tri.neighbors[idx_a]
        c = tri.vertices[idx_c]
        BC = tri.neighbors[idx_b]
        check_hull = (a < 3, b < 3, c < 3)
        if sum(check_hull) == 1:
            if check_hull[0]:
                hull_edges[c] = b
            elif check_hull[1]:
                hull_edges[a] = c
            else:
                hull_edges[b] = a
        branches = [(new_edge, nb_idx) for new_edge, nb_idx in
                    (((a, b), AB), ((b, c), BC)) if nb_idx not in tri_visited]
        for new_edge, nb_idx in branches:
            yield tuple(vertmap[new_edge,])
            if nb_idx not in tri_visited and nb_idx != cdt.NO_NEIGHBOR:
                yield from edges_from_tri(frozenset(new_edge), nb_idx)

    # TODO: Not sure if the starting triangle actually matters.
    #       Maybe it is ok to just remove the for-loop and use
    #       tri = triangles[0], tri_idx = 0
    for tri_idx, tri in enumerate(triangles):
        # make sure to start with a triangle not on the edge of supertriangle
        if cdt.NO_NEIGHBOR not in tri.neighbors:
            break
    edge_start = tuple(vertmap[tri.vertices[:2]])
    ebunch = [edge_start]
    ebunch.extend(edges_from_tri(frozenset(edge_start), tri_idx))
    assert len(tri_visited) == len(triangles)
    # Convert a sequence of hull edges into a sequence of hull nodes
    start, fwd = hull_edges.popitem()
    convex_hull = [vertmap[start]]
    while fwd != start:
        convex_hull.append(vertmap[fwd])
        fwd = hull_edges[fwd]
    return ebunch, convex_hull


def planar_from_cdt_triangles(triangles: list[cdt.Triangle],
                              vertmap: np.ndarray) -> nx.PlanarEmbedding:
    '''
    `triangles` is a `PythonCDT.Triangulation().triangles` list

    `vertmap` is a node number translation table, from CDT numbers to NetworkX

    Returns:
    planar embedding
    '''
    nodes_todo = {}
    nodes_done = set()
    P = nx.PlanarEmbedding()
    tri = triangles[0]
    tri_visited = {0}
    # add the first three nodes to process
    for pivot in tri.vertices:
        nodes_todo[pivot] = 0
    while nodes_todo:
        pivot, tri_idx_start = nodes_todo.popitem()
        tri = triangles[tri_idx_start]
        pivot_idx = tri.vertices.index(pivot)
        succ_start = tri.vertices[(pivot_idx + 1) % 3]
        nb_idx_start_reverse = (pivot_idx - 1) % 3
        succ_end = tri.vertices[(pivot_idx - 1) % 3]
        # first half-edge from `pivot`
        # print('INIT', *vertmap[[pivot, succ_start]])
        P.add_half_edge(*vertmap[[pivot, succ_start]])
        nb_idx = pivot_idx
        ref = succ_start
        refer_to = 'ccw'
        cw = True
        while True:
            tri_idx = tri.neighbors[nb_idx]
            if tri_idx == cdt.NO_NEIGHBOR:
                if cw:
                    # revert direction
                    cw = False
                    refer_to = 'cw'
                    # print('REVE', *vertmap[[pivot, succ_end, ref]], refer_to)
                    P.add_half_edge(*vertmap[[pivot, succ_end]],
                                    **{refer_to: vertmap[succ_start]})
                    ref = succ_end
                    tri = triangles[tri_idx_start]
                    nb_idx = nb_idx_start_reverse
                    continue
                else:
                    break
            tri = triangles[tri_idx]
            pivot_idx = tri.vertices.index(pivot)
            succ = (tri.vertices[(pivot_idx + 1) % 3]
                    if cw else
                    tri.vertices[(pivot_idx - 1) % 3])
            nb_idx = pivot_idx if cw else (pivot_idx - 1) % 3
            # print('NORM', *vertmap[[pivot, succ, ref]], refer_to)
            P.add_half_edge(*vertmap[[pivot, succ]],
                            **{refer_to: vertmap[ref]})
            if succ not in nodes_todo and succ not in nodes_done:
                nodes_todo[succ] = tri_idx
            if succ == succ_end:
                break
            ref = succ
        nodes_done.add(pivot)
    return P


def hull_processor(P: nx.PlanarEmbedding, N: int,
                   supertriangle: tuple[int, int, int],
                   vertex2conc_map: dict[int, int]) \
        -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    '''
    Iterates over the edges that form a triangle with one of supertriangle's
    vertices.

    The supertriangle vertices must have indices in `range(N + B - 3, N + B)`

    If the border has concavities, `to_remove` will be non-empty.

    Multiple goals:
        - Get the node sequence that form the convex hull
        - Get the edges that enable a path to go around the outside of a
          concavity exclusion zone.

    Returns:
    convex_hull
    to_remove
    '''
    a, b, c = supertriangle
    convex_hull = []
    conc_outer_edges = []
    to_remove = []
    for pivot, begin, end in ((a, c, b),
                              (b, a, c),
                              (c, b, a)):
        print('==== pivot', pivot, '====')
        source, target = tee(P.neighbors_cw_order(pivot))
        outer = begin
        for u, v in zip(source, chain(target, (next(target),))):
            if u >= N and v >= N:
                if u == outer:
                    to_remove.append((pivot, u))
                    print('del_sup', pivot, u)
                    outer = v
                elif v == end:
                    to_remove.append((pivot, v))
                    print('del_sup', pivot, v)
                if vertex2conc_map.get(u, -1) == vertex2conc_map.get(v, -2):
                    to_remove.append((u, v))
                    conc_outer_edges.append((u, v))
                    print('del_int', u, v)
                    outer = v
            if u != begin and u != end and v != end:
                # if u is not in supertriangle, it is convex_hull
                convex_hull.append(u)
    return convex_hull, to_remove, conc_outer_edges


def flip_triangles_near_exclusions(P: nx.PlanarEmbedding, N: int, B: int,
                                   VertexC: np.ndarray) \
        -> list[tuple[tuple[int, int]]]:
    '''
    Changes P in-place.
    '''
    changes = {}
    border_nodes = set(range(N, N + B - 3)) & P.nodes
    while border_nodes:
        u = border_nodes.pop()
        nbcw = P.neighbors_cw_order(u)
        rev = next(nbcw)
        cur = next(nbcw)
        for fwd in chain(nbcw, (rev, cur)):
            print('looking at:', F[u], F[cur], end=' | ')
            if ((rev < N)
                    and (fwd < N)
                    and (u, cur) in P.edges
                    and not (cur, u) in changes
                    and not is_same_side(*VertexC[[rev, fwd, u, cur]])
                    and P[rev][u]['ccw'] == cur
                    and P[fwd][u]['cw'] == cur):
                print('changing to:', F[rev], F[fwd])
                changes[(u, cur)] = rev, fwd
                P.remove_edge(u, cur)
                P.add_half_edge(rev, fwd, cw=u)
                P.add_half_edge(fwd, rev, cw=cur)
                border_nodes.add(u)
                break
            rev = cur
            cur = fwd
    print('')
    return changes


def flip_triangles_exclusions_super(P: nx.PlanarEmbedding, N: int, B: int,
                                    VertexC: np.ndarray, max_tri_AR: float) \
        -> list[tuple[tuple[int, int]]]:
    '''
    Changes P in-place.

    Some flat triangles may be created within border vertices. These might
    block the clearing of portals that go around concavities (performed by
    `hull_processor()`). This function flips these triangles so that they have
    a vertex in the supertriangle.
    '''
    changes = {}
    idx_ST = N + B - 3
    print('idx_ST', idx_ST)
    # examine only border triangles
    for u in range(N, idx_ST):
        if u not in P.nodes:
            continue
        nbcw = P.neighbors_cw_order(u)
        rev = next(nbcw)
        cur = next(nbcw)
        for fwd in chain(nbcw, (rev, cur)):
            print('looking at:', F[u], F[cur], end=' | ')
            if (cur < idx_ST and (
                    ((N <= rev < idx_ST) and fwd >= idx_ST
                     and triangle_AR(*VertexC[[u, cur, rev]]) > max_tri_AR)
                    or ((N <= fwd < idx_ST) and rev >= idx_ST
                        and triangle_AR(*VertexC[[u, cur, fwd]]) > max_tri_AR))
                    and P[rev][u]['ccw'] == cur
                    and P[fwd][u]['cw'] == cur):
                print('changing to:', F[rev], F[fwd])
                changes[(u, cur)] = rev, fwd
                P.remove_edge(u, cur)
                P.add_half_edge(rev, fwd, cw=u)
                P.add_half_edge(fwd, rev, cw=cur)
            rev = cur
            cur = fwd
    print('')
    return changes


def G_with_contours_from_G_topology(G_topo: nx.Graph,
                                    planar: nx.PlanarEmbedding):
    '''
    can we make G_topo's edges contain in its attributes the contour segments
    lengths? (so that we do not need to recalculate distances, since distances
    are only available in P_paths)

    Takes a solution based on a modified A graph and introduces in the G output
    the internal paths of contoured-edges (border touching).
    '''
    return


def make_planar_embedding(
        G: nx.Graph,
        #  M: int, VertexC: np.ndarray,
        #  boundaries: list[np.ndarray] | None = None,
        offset_scale: float = 1e-4,
        max_tri_AR: float = 30) -> \
        tuple[nx.PlanarEmbedding, nx.Graph]:
    '''
    `offset_scale`:
        Fraction of the diagonal of the site's bbox to use as spacing between
        border and nodes in concavities (only where nodes are the border).
    '''

    # ######
    # Steps:
    # ######
    # A) Transform border concavities in polygons.
    # B) Check if concavities' vertices coincide with wtg. Where they do,
    #    create new concavity vertices to the inside of the concavity.
    # C) Get Delaynay triangulation of the wtg+oss nodes only.
    # D) Add concavities+exclusions and get the Constrained Delaunay Triang.
    # E) Build the planar embedding of the constrained triangulation.
    # F) Build the available-edges graph A and its planar embedding.
    # G) Build P_paths.
    # H) Revisit A to replace edges outside of the border by P_path contours.
    # I) Revisit A to update d2roots according to lengths along P_paths.
    # J) Calculate the area of the concave hull.

    VertexC, border, exclusions, M, N, B = (
        G.graph.get(k) for k in ('VertexC', 'border', 'exclusions',
                                 'M', 'N', 'B'))
    points = np.fromiter((gonb.Point(*xy) for xy in VertexC),
                         dtype=object,
                         count=N + B + M)

    # ############################################
    # A) Transform border concavities in polygons.
    # ############################################
    print('\nPART A')
    border_vertice_from_point = {
            point: i for i, point in enumerate(points[N:-M], start=N)}

    # Turn the main border's concave zones into exclusion polygons.
    borderPoly = gonb.Polygon(border=gonb.Contour(points[border]))
    border_convex_hull = borderPoly.convex_hull
    concavityMPoly = border_convex_hull - borderPoly

    hull_border_vertices = []
    for hullpt in border_convex_hull.border.vertices:
        if hullpt in border_vertice_from_point:
            hull_border_vertices.append(border_vertice_from_point[hullpt])

    # ###################################################################
    # B) Check if concavities' vertices coincide with wtg. Where they do,
    #    create new concavity vertices to the inside of the concavity.
    # ###################################################################
    print('\nPART B')
    vertices_to_move = set()
    offset = offset_scale*np.hypot(*(VertexC.max(axis=0)
                                     - VertexC.min(axis=0)))
    #  print(f'offset: {offset}')
    newC = []
    remove_from_border_pt_map = []
    B_old = B
    # container for the concavities after duplicating coinciding vertices
    concavityPolys = []
    for concavityPoly in getattr(concavityMPoly, 'polygons',
                                 (concavityMPoly,)):
        changed = False
        if concavityPoly is gonb.EMPTY:
            continue
        print('concavityPoly', concavityPoly)
        new_coords = []
        new_points = []
        vertices = concavityPoly.border.vertices
        rev = vertices[-1]
        X = border_vertice_from_point[rev]
        X_is_hull = X in hull_border_vertices
        cur = vertices[0]
        Y = border_vertice_from_point[cur]
        Y_is_hull = Y in hull_border_vertices
        for fwd in chain(vertices[1:], (cur,)):
            Z = border_vertice_from_point[fwd]
            Z_is_hull = fwd in border_convex_hull.border.vertices
            if cur in points[:N]:
                # Concavity border vertex coincides with node.
                # Therefore, move the border.
                XY = VertexC[Y] - VertexC[X]
                YZ = VertexC[Z] - VertexC[Y]
                _XY_ = np.hypot(*XY)
                _YZ_ = np.hypot(*YZ)
                nXY = XY[::-1].copy()/_XY_
                nYZ = YZ[::-1].copy()/_YZ_
                nXY[0] = -nXY[0]
                nYZ[0] = -nYZ[0]
                angle = np.arccos(np.dot(-XY, YZ)/_XY_/_YZ_)
                if abs(angle) < np.pi/2:
                    # XYZ acute
                    print('acute', end=' ')
                    # project nXY on YZ
                    proj = YZ/_YZ_/max(0.5, np.sin(abs(angle)))
                else:
                    # XYZ obtuse
                    print('obtuse', end=' ')
                    # project nXY on YZ
                    proj = YZ*np.dot(nXY, YZ)/_YZ_**2
                if Y_is_hull:
                    if X_is_hull:
                        print('XY hull', end=' ')
                        # project nYZ on XY
                        T = offset*(-XY/_XY_/max(0.5, np.sin(angle)))
                    elif Z_is_hull:
                        # project nXY on YZ
                        T = offset*(YZ/_YZ_/max(0.5, np.sin(angle)))
                        print('YZ hull', end=' ')
                else:
                    T = offset*(nYZ+proj)
                print(T, np.hypot(*T))
                new_coord = VertexC[Y] + T
                new_coords.append(new_coord)
                new_point = gonb.Point(*new_coord)
                new_points.append(new_point)
                remove_from_border_pt_map.append(cur)
                border_vertice_from_point[new_point] = N + B
                B += 1
                changed = True
            else:
                new_coords.append(VertexC[Y])
                new_points.append(cur)
            X, X_is_hull = Y, Y_is_hull
            Y, Y_is_hull = Z, Z_is_hull
            Y_is_hull = fwd in border_convex_hull.border.vertices
            cur = fwd
        if changed:
            print('changed')
            concavityPolys.append(
                    gonb.Polygon(border=gonb.Contour(new_points)))
            newC.append(np.array(new_coords))
        else:
            concavityPolys.append(concavityPoly)
    if changed:
        print('newC lengths: ', [len(nc) for nc in newC],
              'B without supertriangle:', B)

    for pt in remove_from_border_pt_map:
        del border_vertice_from_point[pt]

    # ########################################################
    # C) Get Delaynay triangulation of the wtg+oss nodes only.
    # ########################################################
    print('\nPART C')
    vertice_from_point = (
        border_vertice_from_point
        | {point: i for i, point in enumerate(points[:N])}
        | {point: i for i, point in zip(range(-M, 0), points[-M:])}
    )

    if exclusions is not None:
        exclusionsMPoly = gonb.Multipolygon(
            (gonb.Polygon(border=gonb.Contour(points[exc]))
             for exc in exclusions))

    # assemble all points actually used in concavityPoly
    points_used = set()
    for concPoly in concavityPolys:
        points_used.update(set(concPoly.border.vertices))

    # create the PythonCDT vertices
    verticesCDT = []
    verticeCDT_from_point = {}
    vertice_from_verticeCDT = np.empty((3 + N + M + len(points_used),),
                                       dtype=int)
    # account for the supertriangle vertices that cdt.Triangulation() adds
    supertriangle = tuple(range(N + B, N + B + 3))
    B += 3
    vertice_from_verticeCDT[:3] = supertriangle
    for i, pt in enumerate(chain(points[:N], points[-M:], points_used)):
        verticesCDT.append(cdt.V2d(pt.x, pt.y))
        # this one is used before supertriangle (no + 3)
        verticeCDT_from_point[pt] = i
        # this one is used after supertriangle (hence, + 3)
        vertice_from_verticeCDT[i + 3] = vertice_from_point[pt]

    # Create triangulation and add vertices and edges
    mesh = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO,
                             cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    mesh.insert_vertices(verticesCDT[:N + M])

    P_A = planar_from_cdt_triangles(mesh.triangles, vertice_from_verticeCDT)

    # ######################################################################
    # D) Add concavities+exclusions and get the Constrained Delaunay Triang.
    # ######################################################################
    print('\nPART D')
    # create the PythonCDT edges
    edgesCDT = []
    concavityVertexSets = []
    for concPoly in concavityPolys:
        edgesCDT.extend((cdt.Edge(verticeCDT_from_point[seg.start],
                                  verticeCDT_from_point[seg.end])
                         for seg in concPoly.edges))
        concavityVertexSets.append(tuple(vertice_from_point[v]
                                         for v in concPoly.border.vertices))
    # TODO: add exclusion zones

    mesh.insert_vertices(verticesCDT[N + M:])
    mesh.insert_edges(edgesCDT)

    # remove triangles inside holes
    mesh.remove_triangles(
            set(np.flatnonzero(mesh.calculate_triangle_depths())))

    # add any newly created plus the supertriangle's vertices to VertexC
    # note: B has already been increased by all newC lengths within the loop
    VertexC = np.vstack((VertexC[:-M],
                         *newC,
                         [(v.x, v.y)
                          for v in mesh.vertices[:3]],
                         VertexC[-M:]))

    # ###############################################################
    # E) Build the planar embedding of the constrained triangulation.
    # ###############################################################
    print('\nPART E')
    P = planar_from_cdt_triangles(mesh.triangles, vertice_from_verticeCDT)

    concavityVertex2concavity = {}
    #  for conc in concavityVertexSets:
    for concavity_idx, conc in enumerate(concavityVertexSets):
        a, b = tee(conc)
        for u, v in zip(a, chain(b, (next(b),))):
            concavityVertex2concavity[u] = concavity_idx
            print(u, v)
            P[u][v]['kind'] = 'concavity'
            P[v][u]['kind'] = 'concavity'

    P.check_structure()

    # adjust flat triangles around concavities
    changes_super = flip_triangles_exclusions_super(
            P, N, B, VertexC, max_tri_AR=max_tri_AR)

    # make_concavities_intransponible(P, N, B, tuple(range(N + B - 3, N + B)))
    convex_hull, to_remove, conc_outer_edges = hull_processor(
            P, N, supertriangle, concavityVertex2concavity)
    P.remove_edges_from(to_remove)

    # ##############################################################
    # F) Build the available-edges graph A and its planar embedding.
    # ##############################################################
    print('\nPART F')
    #  hull_A, _ = hull_processor(P_A, N, supertriangle, concavityVertex2concavity)
    convex_hull_A = []
    a, b, c = supertriangle
    for pivot, begin, end in ((a, c, b),
                              (b, a, c),
                              (c, b, a)):
        source, target = tee(P_A.neighbors_cw_order(pivot))
        for u, v in zip(source, chain(target, (next(target),))):
            if u != begin and u != end and v != end:
                convex_hull_A.append(u)
    print('convex_hull_A:', '–'.join(F[n] for n in convex_hull_A))

    # Prune flat triangles from A (criterion is aspect_ratio <= `max_tri_AR`).
    # Also create a new hull without the triangles: `hull_prunned`.
    queue = list(zip(convex_hull_A[::-1],
                     chain(convex_hull_A[0:1], convex_hull_A[:0:-1])))
    hull_prunned = []
    while queue:
        u, v = queue.pop()
        n = P_A[u][v]['ccw']
        if triangle_AR(*VertexC[[u, v, n]]) > max_tri_AR:
            P_A.remove_edge(u, v)
            queue.extend(((n, v), (u, n)))
            continue
        hull_prunned.append(u)
    print('hull_prunned:', '–'.join(F[n] for n in hull_prunned))

    P_A_undir = P_A.to_undirected(as_view=True)

    diagonals = {}
    for u, v in ((u, v) for u, v in P_A_undir.edges
                 if (not (u in hull_prunned and v in hull_prunned)
                     and u < N and v < N)):
        s = P_A[u][v]['cw']
        t = P_A[u][v]['ccw']
        if is_triangle_pair_a_convex_quadrilateral(*VertexC[[u, v, s, t]]):
            u, v, n = (s, t, v) if s < t else (t, s, u)
            diagonals[(u, v)] = n

    A = nx.Graph(VertexC=VertexC, border=border, N=N, M=M, B=B - 3,
                 planar=P_A,
                 diagonals=diagonals,
                 name=G.graph['name'],
                 handle=G.graph['handle'],
                 landscape_angle=G.graph['landscape_angle'],
                 # TODO: make these attribute names consistent across the code
                 hull=convex_hull_A,
                 hull_prunned=hull_prunned)
    A.add_edges_from(((u, v) for u, v in P_A_undir.edges if u < N and v < N),
                     kind='delaunay')

    # add the diagonals to A
    diagnodes = np.empty((len(diagonals), 2), dtype=int)
    for row, uv in zip(diagnodes, diagonals):
        row[:] = uv
    A.add_edges_from(diagonals, kind='extended')
    Length = np.hypot(*(VertexC[diagnodes[:, 0]]
                        - VertexC[diagnodes[:, 1]]).T)
    for (u, v), length in zip(diagnodes, Length):
        A[u][v]['length'] = length

    # ### Add length attribute to A's and P's edges
    A_edges = np.array(A.edges)

    A_lengths = np.hypot(*(VertexC[A_edges[:, 1]] - VertexC[A_edges[:, 0]]).T)

    nx.set_edge_attributes(A, {tuple(edge): length
                               for edge, length in
                               zip(A_edges, A_lengths)},
                           name='length')

    # TODO: move this flip and check to a more adequate context
    changes_exclusions = flip_triangles_near_exclusions(P, N, B, VertexC)
    P.check_structure()

    # #################
    # G) Build P_paths.
    # #################
    print('\nPART G')
    P_paths = P.to_undirected()
    P_paths.remove_nodes_from(supertriangle)

    # this adds diagonals to P_paths
    # we don't want diagonals where P_paths[u][v]['kind'] is set ('concavity')
    for u, v in [(u, v) for u, v in P_paths.edges
                 #  if (not (u in hull_A and v in hull_A)
                 if (not (u in hull_prunned and v in hull_prunned)
                     and P_paths[u][v].get('kind') is None)]:
        s = P[u][v]['cw']
        t = P[u][v]['ccw']
        if (s >= N + B - 3 or t >= N + B - 3):
            #  or any((((u in conc) and (v in conc))
            #            for conc in concavityVertices))):
            continue
        if is_triangle_pair_a_convex_quadrilateral(*VertexC[[u, v, s, t]]):
            print(u, v, end=' ')
            u, v, n = (s, t, v) if s < t else (t, s, u)
            print(u, v)
            P_paths.add_edge(u, v, length=np.hypot(*(VertexC[u]
                                                     - VertexC[v]).T))

    nx.set_edge_attributes(P_paths, {tuple(edge): length
                                     for edge, length in
                                     zip(A_edges, A_lengths)},
                           name='length')
    for u, v, eData in P_paths.edges(data=True):
        if 'length' not in eData:
            eData['length'] = np.hypot(*(VertexC[u] - VertexC[v]).T)

    not_in_P = A.edges - P_paths.edges
    cw, ccw = rotation_checkers_factory(VertexC)

    # Use `not_in_P` for obtaining the hull_concave from hull_prunned
    queue = list(zip(hull_prunned[::-1], chain((hull_prunned[0],),
                                               hull_prunned[:0:-1]),))
    hull_concave = []
    print('hull_concave = ', end='')
    while queue:
        u, v = queue.pop()
        if (u, v) in not_in_P or (v, u) in not_in_P:
            #  n = (A[u] & A[v]).pop()
            n = P_A[u][v]['ccw']
            #  candidates = set(A[u]) & set(A[v])
            print(f'({F[n]})', end=' ')
            #  n = ().pop()
            queue.extend(((n, v), (u, n)))
            continue
        hull_concave.append(u)
        print(F[u], end=', ')
    print('')
    A.graph['hull_concave'] = hull_concave

    # #######################################################################
    # H) Revisit A to replace edges outside of the border by P_path contours.
    # #######################################################################
    print('\nPART H')
    corner_to_A_edges = defaultdict(list)
    A_edges_to_revisit = []
    for u, v in not_in_P:
        # For the edges in A that are not in P, we find their corresponding
        # shortest path in P_path and update the length attribute in A.
        print(u, v, end=': ')
        length, path = nx.bidirectional_dijkstra(P_paths, u, v,
                                                 weight='length')
        print(length, path)
        if all(n >= N for n in path[1:-1]):
            # keep only paths that only have border vertices between nodes
            eData = A[path[0]][path[-1]]
            original_path = (path[1:-1].copy()
                             if u < v else
                             path[-2:0:-1].copy())
            i = 0
            while i <= len(path) - 3:
                # Check if each vertice at the border is necessary.
                # The vertice is kept if the border angle and the path angle
                # point to the same side. Otherwise, remove the vertice.
                s, b, t = path[i:i + 3]
                # print(i, b)
                a, c = (n for n in P[b]
                        if (P[b][n].get('kind') == 'concavity'
                            or n in supertriangle))
                a, c = (a, c) if P[a][b]['ccw'] == c else (c, a)
                test = ccw if cw(a, b, s) else cw
                if test(s, b, t):
                    i += 1
                    print(s, b, t, 'passed')
                else:
                    # TODO: The entire new path should go for a 2nd pass if it
                    #       changed here. Unlikely to change in the 2nd pass.
                    #       Reason: a shortcut may change the geometry in such
                    #       way as to make additional shortcuts possible.
                    del path[i + 1]
                    length -= P_paths[s][b]['length'] + P_paths[b][t]['length']
                    length += np.hypot(*(VertexC[s] - VertexC[t]).T)
                    shortcut = eData.get('shortcut')
                    if shortcut is None:
                        eData['shortcut'] = [b]
                    else:
                        shortcut.append(b)
                    print(s, b, t, 'shortcut')
            if len(path) > 2:
                eData.update(length=length,
                             # path-> P edges used to calculate A edge's length
                             # path=path[1:-1],
                             # original_path-> which P edges the A edge maps to
                             # (so that PathFinder works)
                             path=original_path,
                             kind='corner_'+eData['kind'])
                u, v = (u, v) if u < v else (v, u)
                for p in path[1:-1]:
                    corner_to_A_edges[p].append((u, v))
            else:
                eData['path'] = original_path
        else:
            # remove edge because the path goes through some wtg node
            u, v = (u, v) if u < v else (v, u)
            A.remove_edge(u, v)
            if (u, v) in diagonals:
                del diagonals[(u, v)]
            else:
                # Some edges will need revisiting to maybe promote their
                # diagonals to delaunay edges.
                A_edges_to_revisit.append((u, v))
    A.graph['corner_to_A_edges'] = corner_to_A_edges

    for n in range(N):
        A.nodes[n]['kind'] = 'wtg'
    for r in range(-M, 0):
        A.nodes[r]['kind'] = 'oss'

    # Diagonals in A which have a missing origin Delaunay edge become edges.
    for u, v in ((u, v) for u, v in A_edges_to_revisit
                 if (not (u in hull_prunned and v in hull_prunned))):
                 #  if (not (u in hull_A and v in hull_A))):
        s = P_A[u][v]['cw']
        t = P_A[u][v]['ccw']
        s, t = (s, t) if s < t else (t, s)
        if (s, t) in diagonals:
            eData = A[s][t]
            eData['kind'] = ('corner_delaunay'
                             if 'path' in eData else
                             'delaunay')
            del diagonals[(s, t)]

    # ##################################################################
    # I) Revisit A to update d2roots according to lengths along P_paths.
    # ##################################################################
    print('\nPART I')
    # use P_paths to obtain estimates of d2roots taking into consideration the
    # concavities and exclusion zones
    if len(concavityVertexSets) > 0 or (exclusions and len(exclusions) > 0):
        # d2roots may not be the plain Euclidean distance
        # d2roots = np.empty((M, N), dtype=float)
        d2roots = cdist(VertexC[:-(M + B)], VertexC[-M:])
        A.graph['d2roots'] = d2roots
        G.graph['d2roots'] = d2roots
        for r in range(-M, 0):
            lengths, paths = nx.single_source_dijkstra(P_paths, r,
                                                       weight='length')
            for n, path in paths.items():
                if n >= N:
                    continue
                if any(p >= N for p in path):
                    # This estimate may be slightly longer that just going
                    # around the border.
                    print(f'changing {n} with path', path)
                    node_d2roots = A.nodes[n].get('d2roots')
                    if node_d2roots is None:
                        A.nodes[n]['d2roots'] = {r: d2roots[n, r]}
                    else:
                        node_d2roots.update({r: d2roots[n, r]})
                    d2roots[n, r] = lengths[n]

    # ##########################################
    # J) Calculate the area of the concave hull.
    # ##########################################
    bX, bY = VertexC[hull_prunned].T
    lower_bound = np.array((bX.min(), bY.min()), dtype=np.float64)
    upper_bound = np.array((bX.max(), bY.max()), dtype=np.float64)
    # Shoelace formula for area (https://stackoverflow.com/a/30408825/287217).
    # Then take the sqrt() to get the linear scaling factor such that area=1.
    norm_factor = math.sqrt(0.5*(bX[-1]*bY[0] - bY[-1]*bX[0]
                            + np.dot(bX[:-1], bY[1:])
                            - np.dot(bY[:-1], bX[1:])))
    A.graph.update(dict(lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        norm_factor=norm_factor))

    # products:
    # P: PlanarEmbedding
    # A: Graph (carries the updated VertexC)
    # P_A: PlanarEmbedding
    # P_paths: Graph
    # diagonals: dict
    # TODO: remove concavityPolys from returned tuple
    # concavityPolys: list

    # TODO: This block came from deprecated `delaunay()`. Analyse whether to
    #       include part of that here.
    #  if bind2root:
    #      for n, n_root in G_base.nodes(data='root'):
    #          A.nodes[n]['root'] = n_root
    #      # alternatively, if G_base nodes do not have 'root' attr:
    #      #  for n, nodeD in A.nodes(data=True):
    #      #      nodeD['root'] = -M + np.argmin(d2roots[n])
    #      # assign each edge to the root closest to the edge's middle point
    #      for u, v, edgeD in A.edges(data=True):
    #          edgeD['root'] = -M + np.argmin(
    #                  cdist(((VertexC[u] + VertexC[v])/2)[np.newaxis, :],
    #                        VertexC[-M:]))
    #  A.graph.update(
    #                 planar=planar,
    #                 d2roots=d2roots,
    #                 landscape_angle=G_base.graph.get('landscape_angle', 0),
    #                 name=G_base.graph['name'],
    #                 handle=G_base.graph['handle']
    #                )

    # TODO: are `concavityPolys` returned only for debugging?
    return P, A
    #  return P, A, diagonals, concavityPolys


def delaunay(G: nx.Graph):
    # TODO: deprecate the use of delaunay()
    P, A = make_planar_embedding(G)
    return A


def A_graph(G_base, delaunay_based=True, weightfun=None, weight_attr='weight'):
    # TODO: refator to be compatible with interarray.mesh's delaunay()
    '''
    Return the "available edges" graph that is the base for edge search in
    Esau-Williams. If `delaunay_based` is True, the edges are the expanded
    Delaunay triangulation, otherwise a complete graph is returned.

    This function is being kept for backward-compatibility. For the Delaunay
    triangulation, call `delaunay()` directly.
    '''
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


def planar_flipped_by_routeset(G: nx.Graph, *, planar: nx.PlanarEmbedding,
                               diagonals: dict[tuple[int, int], int],
                               relax_boundary: bool = False) \
        -> nx.PlanarEmbedding:
    # TODO: this is not going to work now that G has all the segments
    #       we need to find the end node of a contour and check if that
    #       is in diagonals
    '''
    Returns a modified PlanarEmbedding based on `planar`, where all edges used
    in `G` are edges of the output embedding. For this to work, all non-gate
    edges of `G` must be either edges of `planar` or one of `G`'s
    graph attribute 'diagonals'. In addition, `G` must be free of edge×edge
    crossings.
    '''
    M, N, B, D, VertexC, border, exclusions = (
        G.graph.get(k) for k in ('M', 'N', 'B', 'D', 'VertexC', 'border',
                                 'exclusions'))

    P = planar.copy()
    diagonals_base = diagonals
    #  diagonalsʹ = diagonals.copy()
    #  P.graph['diagonals'] = diagonalsʹ
    for r in range(-M, 0):
        #  for u, v in nx.edge_dfs(G, r):
        for u, v in nx.edge_bfs(G, r):
            # update the planar embedding to include any Delaunay diagonals
            # used in G; the corresponding crossing Delaunay edge is removed
            u, v = (u, v) if u < v else (v, u)
            s = diagonals_base.get((u, v))
            if s is not None:
                t = planar[u][s]['ccw']  # same as P[v][s]['cw']
                if (s, t) in G.edges and s >= 0 and t >= 0:
                    # (u, v) & (s, t) are in G (i.e. a crossing). This means
                    # the diagonal (u, v) is a gate and (s, t) should remain
                    continue
                # examine the two triangles (s, t) belongs to
                crossings = False
                for a, b, c in ((s, t, u), (t, s, v)):
                    # this is for diagonals crossing diagonals
                    d = planar[c][b]['ccw']
                    diag_da = (a, d) if a < d else (d, a)
                    if (d == planar[b][c]['cw']
                            and diag_da in diagonals_base
                            and diag_da[0] >= 0):
                        crossings = crossings or diag_da in G.edges
                    e = planar[a][c]['ccw']
                    diag_eb = (e, b) if e < b else (b, e)
                    if (e == planar[c][a]['cw']
                            and diag_eb in diagonals_base
                            and diag_eb[0] >= 0):
                        crossings = crossings or diag_eb in G.edges
                if crossings:
                    continue
                P.add_half_edge(u, v, ccw=t)
                P.add_half_edge(v, u, ccw=s)
                P.remove_edge(s, t)
                #  del diagonalsʹ[u, v]
                s, t, v = (s, t, v) if s < t else (t, s, u)
                #  diagonalsʹ[s, t] = v
    return P
