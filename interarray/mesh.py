# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
import numpy as np
import networkx as nx
import shapely as shp
from scipy.spatial.distance import cdist
from loguru import logger

from collections import defaultdict
from itertools import chain, tee, combinations

from bidict import bidict
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
from interarray.geometric import is_triangle_pair_a_convex_quadrilateral

trace, debug, info, success, warn, error, critical = (
    logger.trace, logger.debug, logger.info, logger.success,
    logger.warning, logger.error, logger.critical)

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
        -> tuple[list[int], list[tuple[int, int]], set[tuple[int, int]]]:
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
    conc_outer_edges = set()
    to_remove = []
    for pivot, begin, end in ((a, c, b),
                              (b, a, c),
                              (c, b, a)):
        trace('==== pivot', pivot, '====')
        source, target = tee(P.neighbors_cw_order(pivot))
        outer = begin
        for u, v in zip(source, chain(target, (next(target),))):
            if u >= N and v >= N:
                if u == outer:
                    to_remove.append((pivot, u))
                    trace('del_sup', pivot, u)
                    outer = v
                elif v == end:
                    to_remove.append((pivot, v))
                    trace('del_sup', pivot, v)
                if vertex2conc_map.get(u, -1) == vertex2conc_map.get(v, -2):
                    to_remove.append((u, v))
                    conc_outer_edges.add((u, v) if u < v else (v, u))
                    trace('del_int', u, v)
                    outer = v
            if u != begin and u != end and v != end:
                # if u is not in supertriangle, it is convex_hull
                convex_hull.append(u)
    return convex_hull, to_remove, conc_outer_edges


def _flip_triangles_near_exclusions(P: nx.PlanarEmbedding, N: int, B: int,
                                    VertexC: np.ndarray) \
        -> list[tuple[tuple[int, int]]]:
    '''
    DEPRECATED after the forcing of non-contoured A edges to be kept in P

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
            trace('looking at:', F[u], F[cur])
            if ((rev < N)
                    and (fwd < N)
                    and (u, cur) in P.edges
                    and not (cur, u) in changes
                    and not is_same_side(*VertexC[[rev, fwd, u, cur]])
                    and P[rev][u]['ccw'] == cur
                    and P[fwd][u]['cw'] == cur):
                trace('changing to:', F[rev], F[fwd])
                changes[(u, cur)] = rev, fwd
                P.remove_edge(u, cur)
                P.add_half_edge(rev, fwd, cw=u)
                P.add_half_edge(fwd, rev, cw=cur)
                border_nodes.add(u)
                break
            rev = cur
            cur = fwd
    trace('')
    return changes


def _flip_triangles_exclusions_super(P: nx.PlanarEmbedding, N: int, B: int,
                                     VertexC: np.ndarray, max_tri_AR: float) \
        -> list[tuple[tuple[int, int]]]:
    '''
    DEPRECATED after the forcing of non-contoured A edges to be kept in P

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
        S: nx.Graph,
        #  M: int, VertexC: np.ndarray,
        #  boundaries: list[np.ndarray] | None = None,
        offset_scale: float = 1e-4,
        max_tri_AR: float = 30) -> \
        tuple[nx.PlanarEmbedding, nx.Graph]:
    '''
    This does more than the planar embedding. A name change is in order.

    The available edges graph A is arguably the main product.

    `offset_scale`:
        Fraction of the diagonal of the site's bbox to use as spacing between
        border and nodes in concavities (only where nodes are the border).
    '''

    # ######
    # Steps:
    # ######
    # A) Transform border concavities in polygons.
    # B) Check if concavities' vertices coincide with wtg. Where they do,
    #    create stunt concavity vertices to the inside of the concavity.
    # C) Get Delaynay triangulation of the wtg+oss nodes only.
    # D) Build the available-edges graph A and its planar embedding.
    # E) Add concavities+exclusions and get the Constrained Delaunay Triang.
    # F) Build the planar embedding of the constrained triangulation.
    # G) Build P_paths.
    # H) Revisit A to update edges crossing borders by with P_path contours.
    # I) Revisit A to update d2roots according to lengths along P_paths.
    # J) Calculate the area of the concave hull.
    # X) Create hull_concave.

    VertexC, border, exclusions, M, N, B = (
        S.graph.get(k) for k in ('VertexC', 'border', 'exclusions',
                                 'M', 'N', 'B'))
    points = np.fromiter((gonb.Point(*xy) for xy in VertexC),
                         dtype=object,
                         count=N + B + M)

    # ############################################
    # A) Transform border concavities in polygons.
    # ############################################
    debug('PART A')
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
    #    create stunt concavity vertices to the inside of the concavity.
    # ###################################################################
    debug('PART B')
    offset = offset_scale*np.hypot(*(VertexC.max(axis=0)
                                     - VertexC.min(axis=0)))
    #  debug(f'offset: {offset}')
    stuntC = []
    border_stunts = []
    remove_from_border_pt_map = []
    B_old = B
    # replace coinciding vertices with stunts and save concavities here
    concavityPolys = []
    for concavityPoly in getattr(concavityMPoly, 'polygons',
                                 (concavityMPoly,)):
        changed = False
        if concavityPoly is gonb.EMPTY:
            continue
        debug('concavityPoly: {}', concavityPoly)
        stunt_coords = []
        conc_points = []
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
                # Therefore, create a stunt vertex for the border.
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
                    trace('acute')
                    # project nXY on YZ
                    proj = YZ/_YZ_/max(0.5, np.sin(abs(angle)))
                else:
                    # XYZ obtuse
                    trace('obtuse')
                    # project nXY on YZ
                    proj = YZ*np.dot(nXY, YZ)/_YZ_**2
                if Y_is_hull:
                    if X_is_hull:
                        trace('XY hull')
                        # project nYZ on XY
                        T = offset*(-XY/_XY_/max(0.5, np.sin(angle)))
                    elif Z_is_hull:
                        # project nXY on YZ
                        T = offset*(YZ/_YZ_/max(0.5, np.sin(angle)))
                        trace('YZ hull')
                else:
                    T = offset*(nYZ+proj)
                trace('translation: {}', T)
                # to extract stunts' coordinates:
                # stuntsC = VertexC[N + B - len(border_stunts): N + B]
                border_stunts.append(Y)
                stunt_coord = VertexC[Y] + T
                stunt_point = gonb.Point(*stunt_coord)
                stunt_coords.append(stunt_coord)
                conc_points.append(stunt_point)
                remove_from_border_pt_map.append(cur)
                border_vertice_from_point[stunt_point] = N + B
                B += 1
                changed = True
            else:
                conc_points.append(cur)
            X, X_is_hull = Y, Y_is_hull
            Y, Y_is_hull = Z, Z_is_hull
            Y_is_hull = fwd in border_convex_hull.border.vertices
            cur = fwd
        if changed:
            debug('Concavities changed!')
            concavityPolys.append(
                    gonb.Polygon(border=gonb.Contour(conc_points)))
            stuntC.append(np.array(stunt_coords))
        else:
            concavityPolys.append(concavityPoly)
    # Stunts are added to the B range and they should be saved with routesets.
    # Alternatively, one could convert stunts to clones of their primes, but
    # this could create some small interferences between edges.
    if stuntC:
        debug('stuntC lengths: {}; former B: {}; new B: {}',
              [len(nc) for nc in stuntC], B_old, B)

    for pt in remove_from_border_pt_map:
        del border_vertice_from_point[pt]

    # ########################################################
    # C) Get Delaynay triangulation of the wtg+oss nodes only.
    # ########################################################
    debug('PART C')
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

    # ##############################################################
    # D) Build the available-edges graph A and its planar embedding.
    # ##############################################################
    debug('PART D')
    convex_hull_A = []
    a, b, c = supertriangle
    for pivot, begin, end in ((a, c, b),
                              (b, a, c),
                              (c, b, a)):
        # Circles pivot in cw order -> hull becomes ccw order.
        source, target = tee(P_A.neighbors_cw_order(pivot))
        for u, v in zip(source, chain(target, (next(target),))):
            if u != begin and u != end and v != end:
                convex_hull_A.append(u)
    debug('convex_hull_A: {}', '–'.join(F[n] for n in convex_hull_A))
    P_A.remove_nodes_from(supertriangle)

    # Prune flat triangles from P_A (criterion is aspect_ratio > `max_tri_AR`).
    # Also create a `hull_prunned`, a hull without the triangles (ccw order)
    # and a set of prunned hull edges.
    queue = list(zip(convex_hull_A[::-1],
                     chain(convex_hull_A[0:1], convex_hull_A[:0:-1])))
    hull_prunned = []
    hull_prunned_edges = set()
    while queue:
        u, v = queue.pop()
        n = P_A[u][v]['ccw']
        if triangle_AR(*VertexC[[u, v, n]]) > max_tri_AR:
            P_A.remove_edge(u, v)
            queue.extend(((n, v), (u, n)))
            continue
        hull_prunned.append(u)
        hull_prunned_edges.add((u, v) if u < v else (v, u))
    u, v = hull_prunned[0], hull_prunned[-1]
    hull_prunned_edges.add((u, v) if u < v else (v, u))
    debug('hull_prunned: {}', '–'.join(F[n] for n in hull_prunned))
    debug('hull_prunned_edges: {}',
          ','.join(f'{F[u]}–{F[v]}' for u, v in hull_prunned_edges))

    A = P_A.to_undirected()
    nx.set_edge_attributes(A, 'delaunay', name='kind')
    # TODO: ¿do we really need node attr kind? separate with test: node < 0
    nx.set_node_attributes(A, 'wtg', name='kind')
    for r in range(-M, 0):
        A.nodes[r]['kind'] = 'oss'

    # Extend A with diagonals.
    diagonals = bidict()
    for u, v in tuple(A.edges):
        u, v = (u, v) if u < v else (v, u)
        if (u, v) in hull_prunned_edges:
            continue
        uvD = P_A[u][v]
        s, t = uvD['cw'], uvD['ccw']

        # SANITY check (if hull edges were skipped, this should always hold)
        vuD = P_A[v][u]
        assert s == vuD['ccw'] and t == vuD['cw']

        if is_triangle_pair_a_convex_quadrilateral(*VertexC[[u, v, s, t]]):
            s, t = (s, t) if s < t else (t, s)
            diagonals[(s, t)] = (u, v)
            A.add_edge(s, t, kind='extended')
    # Add length attribute to A's edges.
    A_edges = tuple(A.edges)
    source, target = zip(*A_edges)
    # TODO: ¿use d2roots for root-incident edges? probably not worth it
    A_edge_length = dict(
            zip(A_edges, np.hypot(*(VertexC[source,] - VertexC[target,]).T)))
    nx.set_edge_attributes(A, A_edge_length, name='length')

    # D.1) get hull_concave

    # prevent edges that cross the boudaries from going into PlanarEmbedding
    # an exception is made for edges that include a root node
    # TODO: rewrite this loop using gonb instead of shp
    hull_concave = []
    if border is not None:
        singled_nodes = {}
        hull_prunned_poly = shp.Polygon(VertexC[hull_prunned])
        shp.prepare(hull_prunned_poly)
        border_poly = shp.Polygon(VertexC[border])
        shp.prepare(border_poly)
        if not border_poly.covers(hull_prunned_poly):
            hull_stack = hull_prunned[0:1] + hull_prunned[::-1]
            u, v = hull_prunned[-1], hull_stack.pop()
            while hull_stack:
                edge_line = shp.LineString(VertexC[[u, v]])
                #  if (u >= 0 and v >= 0
                #          and not border_poly.covers(edge_line)):
                if not border_poly.covers(edge_line):
                    t = P_A[u][v]['ccw']
                    if t == u:
                        # degenerate case 1
                        singled_nodes[v] = u
                        hull_concave.append(v)
                        t = v
                        v = u
                        u = t
                        continue
                    if t in hull_prunned:
                        # degenerate case 2
                        if t == hull_concave[-2]:
                            singled_nodes[u] = t
                        else:
                            singled_nodes[v] = t
                        hull_concave.append(t)
                        u = t
                        continue
                    hull_stack.append(v)
                    v = t
                else:
                    hull_concave.append(v)
                    u = v
                    v = hull_stack.pop()
    if not hull_concave:
        hull_concave = hull_prunned
    info('hull_concave: {}', '–'.join(F[n] for n in hull_concave))

    # ######################################################################
    # E) Add concavities+exclusions and get the Constrained Delaunay Triang.
    # ######################################################################
    debug('PART E')
    # create the PythonCDT edges
    constraint_edges = set()
    edgesCDT_P_A = []

    # Add A's hull as constraint edges to ensure A's edges remain in P.
    for s, t in zip(hull_concave, hull_concave[1:] + [hull_concave[0]]):
        constraint_edges.add((s, t) if s < t else (t, s))
        edgesCDT_P_A.append(cdt.Edge(s if s >= 0 else N + M + s,
                                     t if t >= 0 else N + M + t))

    # This ensures P_A edges within the concave hull remain unaltered.
    mesh.insert_edges(edgesCDT_P_A)
    # remove triangles outside the concave_hull of P_A
    #  print(mesh.calculate_triangle_depths())
    #  mesh.remove_triangles(set(np.flatnonzero(
    #      ~np.array(mesh.calculate_triangle_depths(), dtype=bool))))
    #  num_triangles_P_A = mesh.triangles_count()
    #  print('num_triangles_P_A', num_triangles_P_A)

    edgesCDT = []
    concavityVertexSeqs = []
    for concPoly in concavityPolys:
        for seg in concPoly.edges:
            s, t = vertice_from_point[seg.start], vertice_from_point[seg.end]
            st = (s, t) if s < t else (t, s)
            if st in constraint_edges:
                continue
            constraint_edges.add(st)
            edgesCDT.append(cdt.Edge(verticeCDT_from_point[seg.start],
                                     verticeCDT_from_point[seg.end]))
        concavityVertexSeqs.append(tuple(vertice_from_point[v]
                                         for v in concPoly.border.vertices))
    # TODO: add exclusion zones

    mesh.insert_vertices(verticesCDT[N + M:])
    mesh.insert_edges(edgesCDT)
    #  print('\n\n', mesh.calculate_triangle_depths(), '\n\n')

    # this will remove all internal triangles, both in concavities and in the
    # concave hull
    #  mesh.remove_triangles(
    #          set(np.flatnonzero(mesh.calculate_triangle_depths())))

    # TODO: remove triangles inside exclusion zones (depth = 2)
    #  mesh.remove_triangles(
    #          set(np.flatnonzero(mesh.calculate_triangle_depths())))

    # add any newly created plus the supertriangle's vertices to VertexC
    # note: B has already been increased by all stuntC lengths within the loop
    supertriangleC = np.array([(v.x, v.y) for v in mesh.vertices[:3]])
    VertexC = np.vstack((VertexC[:-M],
                         *stuntC,
                         supertriangleC,
                         VertexC[-M:]))

    # ###############################################################
    # F) Build the planar embedding of the constrained triangulation.
    # ###############################################################
    debug('PART F')
    P = planar_from_cdt_triangles(mesh.triangles, vertice_from_verticeCDT)

    concavityVertex2concavity = {}
    for concavity_idx, conc in enumerate(concavityVertexSeqs):
        for rev, cur, fwd in zip(chain((conc[-1],), conc[:-1]),
                                 conc, chain(conc[1:], (conc[0],))):
            concavityVertex2concavity[cur] = concavity_idx
            # Remove edges inside the concavities
            while P[cur][fwd]['ccw'] != rev:
                P.remove_edge(cur, P[cur][fwd]['ccw'])
            P[cur][fwd]['kind'] = P[fwd][cur]['kind'] = 'concavity'

    # adjust flat triangles around concavities
    #  changes_super = _flip_triangles_exclusions_super(
    #          P, N, B + 3, VertexC, max_tri_AR=max_tri_AR)

    convex_hull, to_remove, conc_outer_edges = hull_processor(
            P, N, supertriangle, concavityVertex2concavity)
    P.remove_edges_from(to_remove)
    constraint_edges -= conc_outer_edges
    P.graph.update(M=M, N=N, B=B,
                   constraint_edges=constraint_edges,
                   supertriangleC=supertriangleC,)

    #  changes_exclusions = _flip_triangles_near_exclusions(P, N, B + 3,
    #                                                       VertexC)
    #  P.check_structure()
    #  print('changes_super', [(F[a], F[b]) for a, b in changes_super])
    #  print('changes_exclusions',
    #        [(F[a], F[b]) for a, b in changes_exclusions])

    #  print('&'*80 + '\n', P_A.edges - P.edges, '\n' + '&'*80)
    #  print('\n' + '&'*80)
    #
    #  # Favor the triagulation in P_A over the one in P where possible.
    #  for u, v in P_A.edges - P.edges:
    #      print(F[u], F[v])
    #      s, t = P_A[u][v]['cw'], P_A[u][v]['ccw']
    #      if (s == P_A[v][u]['ccw']
    #              and t == P_A[v][u]['cw']
    #              and (s, t) in P.edges):
    #          w, x = P[s][t]['cw'], P[s][t]['ccw']
    #          if (w == u and x == v
    #                  and w == P[t][s]['ccw']
    #                  and x == P[t][s]['cw']):
    #              print(F[u], F[v], 'replaces', F[s], F[t])
    #              P.add_half_edge(u, v, ccw=t)
    #              P.add_half_edge(v, u, ccw=s)
    #              P.remove_edge(s, t)
    #      #  else:
    #      #      print(F[u], F[v], 'not in P, but', F[s], F[t],
    #      #            'not available for flipping')
    #  print('&'*80)

    # #################
    # G) Build P_paths.
    # #################
    debug('PART G')
    P_paths = P.to_undirected()
    P_paths.remove_nodes_from(supertriangle)

    # this adds diagonals to P_paths
    # we don't want diagonals where P_paths[u][v]['kind'] is set ('concavity')
    for u, v in [(u, v) for u, v, kind in P_paths.edges(data='kind')
                 if (((u, v) if u < v else (v, u)) not in hull_prunned_edges
                     and kind != 'concavity')]:
        uvD = P[u][v]
        s, t = uvD['cw'], uvD['ccw']
        if s >= N + B or t >= N + B:
            # do not add diagonals incident to supertriangle's vertices
            continue
        if is_triangle_pair_a_convex_quadrilateral(*VertexC[[u, v, s, t]]):
            P_paths.add_edge(s, t)

    nx.set_edge_attributes(P_paths, A_edge_length, name='length')
    #  for u, v in P_paths.edges - A_edge_length:
    for u, v, edgeD in P_paths.edges(data=True):
        if 'length' not in edgeD:
            edgeD['length'] = np.hypot(*(VertexC[u] - VertexC[v]))

    # #######################
    # X) Create hull_concave.
    # #######################
    in_A_not_in_P = A.edges - P_paths.edges
    cw, ccw = rotation_checkers_factory(VertexC)

    # Use `in_A_not_in_P` for obtaining the hull_concave from hull_prunned
    queue = list(zip(hull_prunned[::-1], chain((hull_prunned[0],),
                                               hull_prunned[:0:-1]),))
    hull_concave = []
    while queue:
        u, v = queue.pop()
        if (u, v) in in_A_not_in_P or (v, u) in in_A_not_in_P:
            n = P_A[u][v]['ccw']
            queue.extend(((n, v), (u, n)))
            continue
        hull_concave.append(u)
    info('hull_concave: {}', '–'.join(F[n] for n in hull_concave))
    A.graph['hull_concave'] = hull_concave

    # ######################################################################
    # H) Revisit A to update edges crossing borders by with P_path contours.
    # ######################################################################
    debug('PART H')
    corner_to_A_edges = defaultdict(list)
    A_edges_to_revisit = []
    for u, v in in_A_not_in_P:
        # For the edges in A that are not in P, we find their corresponding
        # shortest path in P_path and update the length attribute in A.
        trace('{}–{}', u, v)
        length, path = nx.bidirectional_dijkstra(P_paths, u, v,
                                                 weight='length')
        trace('length: {}; path: {}', length, path)
        if all(n >= N for n in path[1:-1]):
            # keep only paths that only have border vertices between nodes
            edgeD = A[path[0]][path[-1]]
            original_path = (path[1:-1].copy()
                             if u < v else
                             path[-2:0:-1].copy())
            i = 0
            while i <= len(path) - 3:
                # Check if each vertice at the border is necessary.
                # The vertice is kept if the border angle and the path angle
                # point to the same side. Otherwise, remove the vertice.
                s, b, t = path[i:i + 3]
                a, c = (n for n in P[b]
                        if (P[b][n].get('kind') == 'concavity'
                            or n in supertriangle))
                a, c = (a, c) if P[a][b]['ccw'] == c else (c, a)
                test = ccw if cw(a, b, s) else cw
                if test(s, b, t):
                    i += 1
                    trace('({}) {} {} {} passed', i, s, b, t)
                else:
                    # TODO: Bomb-proof this shortcut test. (not robust as-is)
                    # TODO: The entire new path should go for a 2nd pass if it
                    #       changed here. Unlikely to change in the 2nd pass.
                    #       Reason: a shortcut may change the geometry in such
                    #       way as to make additional shortcuts possible.
                    del path[i + 1]
                    length -= P_paths[s][b]['length'] + P_paths[b][t]['length']
                    shortcut_length = np.hypot(*(VertexC[s] - VertexC[t]).T)
                    length += shortcut_length
                    P_paths.add_edge(s, t, length=shortcut_length)
                    shortcut = edgeD.get('shortcut')
                    if shortcut is None:
                        edgeD['shortcut'] = [b]
                    else:
                        shortcut.append(b)
                    trace('({}) {} {} {} shortcut', i, s, b, t)
            if len(path) > 2:
                edgeD.update(length=length,
                             # path-> P edges used to calculate A edge's length
                             # path=path[1:-1],
                             # original_path-> which P edges the A edge maps to
                             # (so that PathFinder works)
                             path=original_path,
                             kind='contour_'+edgeD['kind'])
                u, v = (u, v) if u < v else (v, u)
                for p in path[1:-1]:
                    corner_to_A_edges[p].append((u, v))
            else:
                edgeD['path'] = original_path
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

    # Diagonals in A which have a missing origin Delaunay edge become edges.
    for u, v in (uv for uv in A_edges_to_revisit
                 if uv not in hull_prunned_edges):
        s = P_A[u][v]['cw']
        t = P_A[u][v]['ccw']
        s, t = (s, t) if s < t else (t, s)
        if (s, t) in diagonals:
            edgeD = A[s][t]
            edgeD['kind'] = ('contour_delaunay'
                             if 'path' in edgeD else
                             'delaunay')
            del diagonals[(s, t)]

    # ##################################################################
    # I) Revisit A to update d2roots according to lengths along P_paths.
    # ##################################################################
    debug('PART I')
    d2roots = cdist(VertexC[:N + B + 3], VertexC[-M:])
    # d2roots may not be the plain Euclidean distance if there are obstacles.
    if len(concavityVertexSeqs) > 0 or (exclusions and len(exclusions) > 0):
        # Use P_paths to obtain estimates of d2roots taking into consideration
        # the concavities and exclusion zones.
        for r in range(-M, 0):
            lengths, paths = nx.single_source_dijkstra(P_paths, r,
                                                       weight='length')
            for n, path in paths.items():
                if n >= N:
                    continue
                if any(p >= N for p in path):
                    # This estimate may be slightly longer that just going
                    # around the border.
                    trace(f'changing {n} with path', path)
                    node_d2roots = A.nodes[n].get('d2roots')
                    if node_d2roots is None:
                        A.nodes[n]['d2roots'] = {r: d2roots[n, r]}
                    else:
                        node_d2roots.update({r: d2roots[n, r]})
                    d2roots[n, r] = lengths[n]

    # ##########################################
    # J) Calculate the area of the concave hull.
    # ##########################################
    bX, bY = VertexC[hull_concave].T
    # assuming that coordinates are UTM -> min() as bbox's offset to origin
    norm_offset = np.array((bX.min(), bY.min()), dtype=np.float64)
    # Shoelace formula for area (https://stackoverflow.com/a/30408825/287217).
    # Then take the sqrt() and invert for the linear factor such that area=1.
    norm_scale = 1.0/math.sqrt(0.5*(bX[-1]*bY[0] - bY[-1]*bX[0]
                               + np.dot(bX[:-1], bY[1:])
                               - np.dot(bY[:-1], bX[1:])))

    # Set A's graph attributes.
    A.graph.update(
        VertexC=VertexC, border=border, N=N, M=M, B=B,
        name=S.name,
        handle=S.graph['handle'],
        landscape_angle=S.graph['landscape_angle'],
        planar=P_A,
        diagonals=diagonals,
        d2roots=d2roots,
        # TODO: make these 2 attribute names consistent across the code
        hull=convex_hull_A,
        hull_prunned=hull_prunned,
        hull_concave=hull_concave,
        # experimental attr
        border_stunts=border_stunts,
        norm_offset=norm_offset,
        norm_scale=norm_scale,
    )

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

    return P, A


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


def _deprecated_planar_flipped_by_routeset(
        G: nx.Graph, *, A: nx.Graph, planar: nx.PlanarEmbedding) \
        -> nx.PlanarEmbedding:
    '''
    DEPRECATED

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
    diagonals = A.graph['diagonals']
    P_A = A.graph['planar']
    seen_endpoints = set()
    for u, v in G.edges - P.edges:
        # update the planar embedding to include any Delaunay diagonals
        # used in G; the corresponding crossing Delaunay edge is removed
        u, v = (u, v) if u < v else (v, u)
        if u >= N:
            # we are in a redundant segment of a multi-segment path
            continue
        if v >= N and u not in seen_endpoints:
            uvA = G[u][v]['A_edge']
            seen_endpoints.add(uvA[0] if uvA[1] == u else uvA[1])
            print('path_uv:', F[u], F[v], '->', F[uvA[0]], F[uvA[1]])
            u, v = uvA if uvA[0] < uvA[1] else uvA[::-1]
            path_uv = [u] + A[u][v]['path'] + [v]
            # now ⟨u, v⟩ represents the corresponding edge in A
        else:
            path_uv = None
        st = diagonals.get((u, v))
        if st is not None:
            # ⟨u, v⟩ is a diagonal of Delaunay edge ⟨s, t⟩
            s, t = st
            path_st = A[s][t].get('path')
            if path_st is not None:
                # pick a proxy segment for checking existance of path in G
                source, target = (s, t) if s < t else (t, s)
                st = source, path_st[0]
                path_st = [source] + path_st + [target]
                # now st represents a corresponding segment in G of A's ⟨s, t⟩
            if st in G.edges and s >= 0:
                if u >= 0:
                    print('ERROR: both Delaunay st and diagonal uv are in G, '
                          'but uv is not gate. Edge×edge crossing!')
                # ⟨u, v⟩ & ⟨s, t⟩ are in G (i.e. a crossing). This means
                # the diagonal ⟨u, v⟩ is a gate and ⟨s, t⟩ should remain
                continue
            if u < 0:
                # uv is a gate: any diagonals crossing it should prevail.
                # ensure u–s–v–t is ccw
                u, v = ((u, v)
                        if (P_A[u][t]['cw'] == s
                            and P_A[v][s]['cw'] == t) else
                        (v, u))
                # examine the two triangles ⟨s, t⟩ belongs to
                crossings = False
                for a, b, c in ((s, t, u), (t, s, v)):
                    # this is for diagonals crossing diagonals
                    d = planar[c][b]['ccw']
                    diag_da = (a, d) if a < d else (d, a)
                    if (d == planar[b][c]['cw']
                            and diag_da in diagonals
                            and diag_da[0] >= 0):
                        path_da = A[d][a].get('path')
                        if path_da is not None:
                            diag_da = ((d if d < a else a), path_da[0])
                        crossings = crossings or diag_da in G.edges
                    e = planar[a][c]['ccw']
                    diag_eb = (e, b) if e < b else (b, e)
                    if (e == planar[c][a]['cw']
                            and diag_eb in diagonals
                            and diag_eb[0] >= 0):
                        path_eb = A[e][b].get('path')
                        if path_eb is not None:
                            diag_eb = ((e if e < b else b), path_eb[0])
                        crossings = crossings or diag_eb in G.edges
                if crossings:
                    continue
            # ⟨u, v⟩ is not crossing any edge in G
            # TODO: THIS NEEDS CHANGES: use paths
            #       it gets really complicated if the paths overlap!
            if path_st is None:
                P.remove_edge(s, t)
            else:
                for s, t in zip(path_st[:-1], path_st[1:]):
                    P.remove_edge(s, t)
            if path_uv is None:
                P.add_half_edge(u, v, ccw=s)
                P.add_half_edge(v, u, ccw=t)
            else:
                for u, v in zip(path_uv[:-1], path_uv[1:]):
                    P.add_half_edge(u, v, ccw=s)
                    P.add_half_edge(v, u, ccw=t)
    return P


def planar_flipped_by_routeset(
        G: nx.Graph, *, planar: nx.PlanarEmbedding) \
        -> nx.PlanarEmbedding:
    '''
    Copies `planar` and flips some of its edges so that the returned
    PlanarEmbedding includes all edges present in `G`.

    For this to work, all non-gate edges of `G` must be either edges of
    `planar` or one of `G`'s graph attribute 'diagonals'. In addition, `G`
    must be free of edge×edge crossings.
    '''
    M, N, B, C, D = (G.graph.get(k, 0) for k in ('M', 'N', 'B', 'C', 'D'))
    VertexC, border, exclusions, fnT = (
        G.graph.get(k) for k in ('VertexC', 'border', 'exclusions', 'fnT'))
    if fnT is None:
        fnT = np.arange(M + N + B + C + D)
        fnT[-M:] = range(-M, 0)

    P = planar.copy()
    seen_endpoints = set()
    debug('differences between G and P:')
    for u, v in G.edges - planar.edges:
        u_, v_ = fnT[u], fnT[v]
        if (u_, v_) in planar.edges:
            continue
        debug('{}–{} ({}–{})', u, v, u_, v_)
        intersection = set(planar[u_]) & set(planar[v_])
        if len(intersection) < 2:
            debug('share {} neighbors.', len(intersection))
            continue
        diagonal_found = False
        for s_, t_ in combinations(intersection, 2):
            if ((s_, t_) in planar.edges
                and is_triangle_pair_a_convex_quadrilateral(
                    *VertexC[[s_, t_, u_, v_]])):
                diagonal_found = True
                break
        if not diagonal_found:
            warn('Failed to find flippable for non-planar {}–{}', u_, v_)
            continue
        if s_ >= N:
            s_clones = G.nodes[s_].get('clones', [s_])
            if len(s_clones) > 1:
                warn('s_clones > 1: {} -> {}', s_, s_clones)
            s = s_clones[0]
        else:
            s = s_
        if t_ >= N:
            t_clones = G.nodes[t_].get('clones', [t_])
            if len(t_clones) > 1:
                warn('t_clones > 1: {} -> {}', t_, t_clones)
            t = t_clones[0]
        else:
            t = t_

        if (s, t) in G.edges and (u < 0 or v < 0):
            # not replacing edge with gate
            continue
        if planar[u_][s_]['ccw'] == t_ and planar[v_][t_]['ccw'] == s_:
            pass
        elif planar[u_][s_]['cw'] == t_ and planar[v_][t_]['cw'] == s_:
            s_, t_ = t_, s_
        else:
            warn('{}–{}–{}–{} is not in two triangles.', u_, s_, v_, t_)
            continue
        #  if not (s == planar[v][u]['ccw']
        #          and t == planar[v][u]['cw']):
        #      print(f'{F[u]}–{F[v]} is not in two triangles')
        #      continue
        #  if (s, t) not in planar:
        #      print(f'{F[s]}–{F[t]} is not in planar')
        #      continue
        #  print(f'flipping {F[s_]}–{F[t_]} to {F[u_]}–{F[v_]}')
        P.remove_edge(s_, t_)
        P.add_half_edge(u_, v_, cw=s_)
        P.add_half_edge(v_, u_, cw=t_)
    return P
