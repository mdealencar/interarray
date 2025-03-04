# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

from collections import defaultdict
from itertools import chain, tee, combinations

from bidict import bidict
from ground.base import get_context
from gon.base import (
    Point, Segment, Contour, Polygon, Relation, Location, EMPTY
)
from orient.planar import (
    point_in_polygon, contour_in_region, segment_in_region
)

import PythonCDT as cdt

from .geometric import (
    halfedges_from_triangulation,
    is_crossing_no_bbox,
    triangle_AR,
    is_triangle_pair_a_convex_quadrilateral,
    is_same_side,
    rotation_checkers_factory,
    assign_root,
    area_from_polygon_vertices,
    find_edges_bbox_overlaps,
    apply_edge_exemptions,
    complete_graph,
)
from . import MAX_TRIANGLE_ASPECT_RATIO, info, debug, warn
from .interarraylib import NodeTagger
from .geometric import is_triangle_pair_a_convex_quadrilateral

F = NodeTagger()
NULL = np.iinfo(int).min


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


def planar_from_cdt_triangles(mesh: cdt.Triangulation,
        vertmap: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], set]:
    '''Convert from a PythonCDT.Triangulation to NetworkX.PlanarEmbedding.

    Used internally in `make_planar_embedding()`. Wraps the numba-compiled `halfedges_from_triangulation()`, which does the intensive work.
    
    Args:
        triangles: `PythonCDT.Triangulation().triangles` list
        vertmap: node number translation table, from CDT numbers to NetworkX

    Returns:
        planar embedding
    '''
    num_tri = mesh.triangles_count()
    triangleI = np.empty((num_tri, 3), dtype=np.int_)
    neighborI = np.empty((num_tri, 3), dtype=np.int_)

    for i, tri in enumerate(mesh.triangles):
        triangleI[i] = vertmap[tri.vertices]
        neighborI[i] = tuple((NULL if n == cdt.NO_NEIGHBOR else n)
                             for n in tri.neighbors)
    # formula for number of triangulation's edges is: 3*V - H - 3
    # H = 3 since CDT's Hull is always the supertriangle
    # and because we count half-edges, use expression × 2 
    num_half_edges = 6*mesh.vertices_count() - 12
    halfedges = np.empty((num_half_edges, 3), dtype=np.int_)
    ref_is_cw_ = np.empty((num_half_edges,), dtype=np.bool_)
    halfedges_from_triangulation(triangleI, neighborI, halfedges, ref_is_cw_)
    edges = set((u.item(), v.item()) for u, v in halfedges[:, :2] if u < v)
    return (halfedges, ref_is_cw_), edges


def P_from_halfedge_pack(halfedge_pack: tuple[np.ndarray, np.ndarray]) \
        -> nx.PlanarEmbedding:
    halfedges, ref_is_cw_ = halfedge_pack
    P = nx.PlanarEmbedding()
    for (u, v, ref), ref_is_cw in zip(halfedges, ref_is_cw_):
        if ref == NULL:
            P.add_half_edge(u.item(), v.item())
        else:
            P.add_half_edge(u.item(), v.item(),
                            **{('cw' if ref_is_cw else 'ccw'): ref.item()})
    return P


def hull_processor(P: nx.PlanarEmbedding, T: int,
                   supertriangle: tuple[int, int, int],
                   vertex2conc_id_map: dict[int, int]) \
        -> tuple[list[int], list[tuple[int, int]], set[tuple[int, int]]]:
    '''
    Iterates over the edges that form a triangle with one of supertriangle's
    vertices.

    The supertriangle vertices must have indices in `range(T + B - 3, T + B)`

    If the border has concavities, `to_remove` will be non-empty.

    Multiple goals:
      - Get the node sequence that form the convex hull
      - Get the edges that enable a path to go around the outside of an
        obstacle's border.

    Returns:
      The convex hull, P edges to be removed and outer edges of concavities
    '''
    a, b, c = supertriangle
    convex_hull = []
    conc_outer_edges = set()
    to_remove = []
    for pivot, begin, end in ((a, c, b),
                              (b, a, c),
                              (c, b, a)):
        debug('==== pivot %d ====', pivot)
        source, target = tee(P.neighbors_cw_order(pivot))
        outer = begin
        for u, v in zip(source, chain(target, (next(target),))):
            if u >= T and v >= T:
                if u == outer:
                    to_remove.append((pivot, u))
                    debug('del_sup %d %d', pivot, u)
                    outer = v
                elif v == end:
                    to_remove.append((pivot, v))
                    debug('del_sup %d %d', pivot, v)
                if (vertex2conc_id_map.get(u, -1)
                        == vertex2conc_id_map.get(v, -2)):
                    to_remove.append((u, v))
                    conc_outer_edges.add((u, v) if u < v else (v, u))
                    debug('del_int %d %d', u, v)
                    outer = v
            if u != begin and u != end and v != end:
                # if u is not in supertriangle, it is convex_hull
                convex_hull.append(u)
    return convex_hull, to_remove, conc_outer_edges


def _flip_triangles_near_obstacles(P: nx.PlanarEmbedding, T: int, B: int,
                                    VertexC: np.ndarray) \
        -> list[tuple[tuple[int, int]]]:
    '''
    DEPRECATED after the forcing of non-contoured A edges to be kept in P

    Changes P in-place.
    '''
    changes = {}
    border_nodes = set(range(T, T + B - 3)) & P.nodes
    while border_nodes:
        u = border_nodes.pop()
        nbcw = P.neighbors_cw_order(u)
        rev = next(nbcw)
        cur = next(nbcw)
        for fwd in chain(nbcw, (rev, cur)):
            debug('looking at: %s %s', F[u], F[cur])
            if ((rev < T)
                    and (fwd < T)
                    and (u, cur) in P.edges
                    and not (cur, u) in changes
                    and not is_same_side(*VertexC[[rev, fwd, u, cur]])
                    and P[rev][u]['ccw'] == cur
                    and P[fwd][u]['cw'] == cur):
                debug('changing to: %s %s', F[rev], F[fwd])
                changes[(u, cur)] = rev, fwd
                P.remove_edge(u, cur)
                P.add_half_edge(rev, fwd, cw=u)
                P.add_half_edge(fwd, rev, cw=cur)
                border_nodes.add(u)
                break
            rev = cur
            cur = fwd
    return changes


def _flip_triangles_obstacles_super(P: nx.PlanarEmbedding, T: int, B: int,
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
    idx_ST = T + B - 3
    print('idx_ST', idx_ST)
    # examine only border triangles
    for u in range(T, idx_ST):
        if u not in P.nodes:
            continue
        nbcw = P.neighbors_cw_order(u)
        rev = next(nbcw)
        cur = next(nbcw)
        for fwd in chain(nbcw, (rev, cur)):
            print('looking at:', F[u], F[cur], end=' | ')
            if (cur < idx_ST and (
                    ((T <= rev < idx_ST) and fwd >= idx_ST
                     and triangle_AR(*VertexC[[u, cur, rev]]) > max_tri_AR)
                    or ((T <= fwd < idx_ST) and rev >= idx_ST
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


def make_planar_embedding(
        L: nx.Graph,
        #  R: int, VertexC: np.ndarray,
        #  boundaries: list[np.ndarray] | None = None,
        offset_scale: float = 1e-4,
        max_tri_AR: float = MAX_TRIANGLE_ASPECT_RATIO) -> \
        tuple[nx.PlanarEmbedding, nx.Graph]:
    ''' This does more than the planar embedding. A name change is in order.

    The available edges graph `A` is arguably the main product.

    Args:
        L: locations graph
        offset_scale: Fraction of the diagonal of the site's bbox to use as
            spacing between border and nodes in concavities (only where nodes
            are the border).

    Returns:
        P - the planar embedding graph - and A - the available edges graph.
    '''

    # ######
    # Steps:
    # ######
    # A) Scale the coordinates to avoid CDT errors.
    # A.1) Transform border concavities in polygons.
    # B) Check if concavities' vertices coincide with wtg. Where they do,
    #    create stunt concavity vertices to the inside of the concavity.
    # C) Get Delaynay triangulation of the wtg+oss nodes only.
    # D) Build the available-edges graph A and its planar embedding.
    # Y) Handle obstacles
    # E) Add concavities and get the Constrained Delaunay Triang.
    # F) Build the planar embedding of the constrained triangulation.
    # G) Build P_paths.
    # H) Revisit A to update edges crossing borders with P_path contours.
    # I) Revisit A to update d2roots according to lengths along P_paths.
    # J) Calculate the area of the concave hull.
    # X) Create hull_concave.

    R, T, B, VertexCʹ = (L.graph[k] for k in 'R T B VertexC'.split())
    border = L.graph.get('border')
    obstacles = L.graph.get('obstacles', [])

    # #############################################
    # A) Scale the coordinates to avoid CDT errors.
    # #############################################
    # Since the initialization of the Triangulation class is made with only
    # the wtg coordinates, there are cases where the later addition of border
    # coordinates generate an error. (e.g. Horns Rev 3)
    # This is caused by the supertriangle being calculated from the first
    # batch of vertices (wtg only), which turns out to be too small to fit the
    # border polygon.
    # CDT's supertriangle calculation has a fallback reference value of 1.0 for
    # vertex sets that fall within a small area. The way to circunvent the
    # error described above is to scale all coordinates down so that CDT will
    # use the fallback and this fallback is enough to cover the scaled borders.
    mean = VertexCʹ.mean(axis=0)
    scale = 2.*max(VertexCʹ.max(axis=0) - VertexCʹ.min(axis=0))

    VertexC = (VertexCʹ - mean)/scale
    # geometric context init (packages ground, gon)
    context = get_context()
    points = np.fromiter((Point(float(x), float(y)) for x, y in VertexC),
                         dtype=object,
                         count=T + B + R)

    # ##############################################
    # A.1) Transform border concavities in polygons.
    # ##############################################
    debug('PART A')
    if border is None:
        hull_minus_border = EMPTY
        border_vertex_from_point = {}
        roots_outside = []
    else:
        border_poly = Polygon(border=Contour(points[border]))

        # Check if roots are outside the border. If so, extend the border to them.
        roots_outside = [
            r_pt for r_pt in points[-R:]
            if point_in_polygon(r_pt, border_poly) is Location.EXTERIOR]
        #  print('roots_outside', len(roots_outside), roots_outside, border_poly)
        hull = context.points_convex_hull(roots_outside
                                          + border_poly.border.vertices)
        hull_poly = Polygon(border=Contour(hull))

        border_vertex_from_point = {
                point: i for i, point in enumerate(points[T:-R], start=T)}

        hull_border_vertices = [border_vertex_from_point[hullpt]
                                for hullpt in hull
                                if hullpt in border_vertex_from_point]

        # Turn the main border's concave zones into concavity polygons.
        hull_minus_border = hull_poly - border_poly

    concavities = []
    if hull_minus_border is EMPTY:
        assert len(roots_outside) == 0
    elif hasattr(hull_minus_border, 'polygons'):
        # MultiPolygon
        for p in hull_minus_border.polygons:
            if all((point_in_polygon(r_pt, p) is Location.EXTERIOR)
                   for r_pt in roots_outside):
                concavities.append(p.border)
            else:
                border_poly += p
    elif roots_outside:
        # single Polygon in hull_minus_border includes a root
        border_poly = hull_poly
    else:
        # single Polygon is a concavity
        concavities = [hull_minus_border.border]

    # ###################################################################
    # B) Check if concavities' vertices coincide with wtg. Where they do,
    #    create stunt concavity vertices to the inside of the concavity.
    # ###################################################################
    debug('PART B')
    offset = offset_scale*np.hypot(*(VertexC.max(axis=0)
                                     - VertexC.min(axis=0)))
    #  debug(f'offset: {offset}')
    stuntC = []
    stunts_primes = []
    remove_from_border_pt_map = set()
    B_old = B
    # replace coinciding vertices with stunts and save concavities here
    for i, concavity in enumerate(concavities):
        changed = False
        debug('concavity: %s', concavity)
        stunt_coords = []
        conc_points = []
        vertices = concavity.vertices
        rev = vertices[-1]
        X = border_vertex_from_point[rev]
        X_is_hull = X in hull_border_vertices
        cur = vertices[0]
        Y = border_vertex_from_point[cur]
        Y_is_hull = Y in hull_border_vertices
        # X->Y->Z is in ccw direction
        for fwd in chain(vertices[1:], (cur,)):
            Z = border_vertex_from_point[fwd]
            Z_is_hull = fwd in hull_poly.border.vertices
            if cur in points[:T] or cur in points[-R:]:
                # Concavity border vertex coincides with node.
                # Therefore, create a stunt vertex for the border.
                XY = VertexC[Y] - VertexC[X]
                YZ = VertexC[Z] - VertexC[Y]
                _XY_ = np.hypot(*XY)
                _YZ_ = np.hypot(*YZ)
                nXY = XY[::-1]/_XY_
                nYZ = YZ[::-1]/_YZ_
                # normal to XY, pointing inward
                nXY[0] = -nXY[0]
                # normal to YZ, pointing inward
                nYZ[0] = -nYZ[0]
                angle = np.arccos(np.dot(-XY, YZ)/_XY_/_YZ_)
                if abs(angle) < np.pi/2:
                    # XYZ acute
                    debug('acute')
                    # project nXY on YZ
                    proj = YZ/_YZ_/max(0.5, np.sin(abs(angle)))
                else:
                    # XYZ obtuse
                    debug('obtuse')
                    # project nXY on YZ
                    proj = YZ*np.dot(nXY, YZ)/_YZ_**2
                if Y_is_hull:
                    if X_is_hull:
                        debug('XY hull')
                        # project nYZ on XY
                        S = offset*(-XY/_XY_/max(0.5, np.sin(angle)) - nXY)
                    else:
                        assert Z_is_hull
                        # project nXY on YZ
                        S = offset*(YZ/_YZ_/max(0.5, np.sin(angle)) - nYZ)
                        debug('YZ hull')
                else:
                    S = offset*(nYZ+proj)
                debug('translation: %s', S)
                # to extract stunts' coordinates:
                # stuntsC = VertexC[T + B - len(stunts_primes): T + B]
                stunts_primes.append(Y)
                stunt_coord = VertexC[Y] + S
                stunt_point = Point(*(float(sc) for sc in stunt_coord))
                stunt_coords.append(stunt_coord)
                conc_points.append(stunt_point)
                remove_from_border_pt_map.add(cur)
                border_vertex_from_point[stunt_point] = T + B
                B += 1
                changed = True
            else:
                conc_points.append(cur)
            X, X_is_hull = Y, Y_is_hull
            Y, Y_is_hull = Z, Z_is_hull
            Y_is_hull = fwd in hull_poly.border.vertices
            cur = fwd
        if changed:
            debug('Concavities changed!')
            concavities[i] = Contour(conc_points)
            stuntC.append(mean + scale*np.array(stunt_coords))
    # Stunts are added to the B range and they should be saved with routesets.
    # Alternatively, one could convert stunts to clones of their primes, but
    # this could create some small interferences between edges.
    if stuntC:
        debug('stuntC lengths: %s; former B: %d; new B: %d',
              [len(nc) for nc in stuntC], B_old, B)

    for pt in remove_from_border_pt_map:
        del border_vertex_from_point[pt]

    # #############################################
    # B.2) Create a miriad of indices and mappings.
    # #############################################
    debug('PART B.2')

    vertex_from_point = (
        border_vertex_from_point
        | {point: i for i, point in enumerate(points[:T])}
        | {point: i for i, point in zip(range(-R, 0), points[-R:])}
    )

    if roots_outside:
        border = np.array([vertex_from_point[pt] for pt in hull], dtype=int)

    holes = [Contour(points[obstacle]) for obstacle in obstacles]

    # assemble all points actually used in concavities and obstacles
    num_pt_concavities = sum(len(conc.vertices) for conc in concavities)
    num_pt_holes = sum(len(hole.vertices) for hole in holes)

    # account for the supertriangle vertices that cdt.Triangulation() adds
    supertriangle = (T + B, T + B + 1, T + B + 2)
    iCDT = 0
    vertex_from_iCDT = np.full(
        (3 + T + R + num_pt_concavities + num_pt_holes,), NULL,  dtype=int)
    vertex_from_iCDT[:3] = supertriangle
    V2d_nodes = []
    for iCDT, pt in enumerate(chain(points[:T], points[-R:]), start=iCDT):
        V2d_nodes.append(cdt.V2d(pt.x, pt.y))
        vertex_from_iCDT[iCDT + 3] = vertex_from_point[pt]

    # Bundle concavities that share a common point.
    if len(concavities) > 1:
        # multiple concavities -> join concavities with a common vertex
        stack = [(set(conc.vertices), conc.vertices) for conc in concavities]
        # concavityVertexSeqs uses the unjoined concavity polygons' vertices
        concavityVertexSeqs = [tuple(vertex_from_point[p] for p in points)
                               for _, points in stack]
        ready = []
        while stack:
            refset, reflst = stack.pop()
            stable = True
            for iconc, (tstset, tstlst) in enumerate(stack):
                common = refset & tstset
                if common:
                    common, = common
                    iref, itst = reflst.index(common), tstlst.index(common)
                    joined = (reflst[:iref] + tstlst[itst:]
                              + tstlst[:itst] + reflst[iref:])
                    debug('common vertex: %d -> new contour: %s', common, joined)
                    del stack[iconc]
                    stack.append((refset | tstset, joined))
                    stable = False
                    break
            if stable:
                ready.append(reflst)
        concavities = [Contour(vertex_list) for vertex_list in ready]
    elif len(concavities) == 1:
        concavityVertexSeqs = [tuple(vertex_from_point[v]
                                     for v in concavities[0].vertices)]
    else:
        concavityVertexSeqs = []

    vertex2conc_id_map = {vertex_from_point[p]: i
                          for i, hole in enumerate(chain(concavities, holes))
                          for p in hole.vertices}

    # ########################################################
    # C) Get Delaynay triangulation of the wtg+oss nodes only.
    # ########################################################
    debug('PART C')
    # Create triangulation and add vertices and edges
    mesh = cdt.Triangulation(cdt.VertexInsertionOrder.AUTO,
                             cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0)
    mesh.insert_vertices(V2d_nodes)

    P_A_halfedge_pack, P_A_edges = planar_from_cdt_triangles(mesh,
                                                             vertex_from_iCDT)
    P_A = P_from_halfedge_pack(P_A_halfedge_pack)
    P_A_edges.difference_update((u, v) for v in supertriangle for u in P_A[v])

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
    debug('convex_hull_A: %s', '–'.join(F[n] for n in convex_hull_A))
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
        # P_A is a DiGraph, so there are 2 degrees per undirected edge 
        if (P_A.degree[u] > 4 and P_A.degree[v] > 4
                and triangle_AR(*VertexC[[u, v, n]]) > max_tri_AR):
            P_A.remove_edge(u, v)
            queue.extend(((n, v), (u, n)))
            uv = (u, v) if u < v else (v, u)
            P_A_edges.remove(uv)
            continue
        hull_prunned.append(u)
        uv = (u, v) if u < v else (v, u)
        hull_prunned_edges.add(uv)
    u, v = hull_prunned[0], hull_prunned[-1]
    uv = (u, v) if u < v else (v, u)
    hull_prunned_edges.add(uv)
    debug('hull_prunned: %s', '–'.join(F[n] for n in hull_prunned))
    debug('hull_prunned_edges: %s',
          ','.join(f'{F[u]}–{F[v]}' for u, v in hull_prunned_edges))

    A = nx.Graph(P_A_edges)
    nx.set_edge_attributes(A, 'delaunay', name='kind')
    # TODO: ¿do we really need node attr kind? separate with test: node < 0
    nx.set_node_attributes(A, 'wtg', name='kind')
    for r in range(-R, 0):
        A.nodes[r]['kind'] = 'oss'

    # Extend A with diagonals.
    diagonals = bidict()
    for u, v in P_A_edges - hull_prunned_edges:
        uvD = P_A[u][v]
        s, t = uvD['cw'], uvD['ccw']

        # SANITY check (if hull edges were skipped, this should always hold)
        vuD = P_A[v][u]
        assert s == vuD['ccw'] and t == vuD['cw']

        if is_triangle_pair_a_convex_quadrilateral(*VertexC[[u, v, s, t]]):
            s, t = (s, t) if s < t else (t, s)
            diagonals[(s, t)] = (u, v)
            A.add_edge(s, t, kind='extended')

    # D.1) get hull_concave

    # prevent edges that cross the boudaries from going into PlanarEmbedding
    # an exception is made for edges that include a root node
    hull_concave = []
    if border is not None:
        hull_prunned_cont = Contour(points[hull_prunned])
        border_cont = border_poly.border
        hull__border = contour_in_region(hull_prunned_cont, border_cont)
        pushed = 0
        if hull__border in (Relation.CROSS, Relation.TOUCH, Relation.OVERLAP):
            hull_stack = hull_prunned[0:1] + hull_prunned[::-1]
            u, v = hull_prunned[-1], hull_stack.pop()
            while hull_stack:
                edge_seg = Segment(points[u], points[v])
                edge_to_border = segment_in_region(edge_seg, border_cont)
                if (edge_to_border is Relation.CROSS
                        or edge_to_border is Relation.TOUCH):
                    t = P_A[u][v]['ccw']
                    #  print(f'[{pushed}]', F[u], F[v], f'⟨{F[t]}⟩', [F[n] for n in hull_stack[::-1]])
                    if t == u:
                        # degenerate case 1
                        hull_concave.append(v)
                        t, v, u = v, u, t
                        continue
                    pushed += 1
                    hull_stack.append(v)
                    if pushed and not any(n in A[t] for n in hull_stack[-pushed:]):
                        # TODO: figure out how to avoid repeated outlier nodes
                        warn('unable to include in hull_concave: %s',
                             ' '.join(F[n] for n in hull_stack[-pushed:]))
                        hull_outliers = A.graph.get('hull_outliers')
                        if hull_outliers is not None:
                            hull_outliers.extend(hull_stack[-pushed:])
                        else:
                            A.graph['hull_outliers'] = hull_stack[-pushed:]
                        del hull_stack[-pushed:]
                        pushed = 0
                        while hull_stack:
                            v = hull_stack.pop()
                            if v not in hull_concave:
                                break
                        continue
                    v = t
                else:
                    #  print(f'[{pushed}]', F[u], F[v], [F[n] for n in hull_stack[::-1]])
                    hull_concave.append(v)
                    u = v
                    if pushed:
                        pushed -= 1
                    v = hull_stack.pop()
    if not hull_concave:
        hull_concave = hull_prunned
    debug('hull_concave: %s', '–'.join(F[n] for n in hull_concave))

    # ######################################################################
    # Y) Handle obstacles
    # ######################################################################
    debug('PART Y')
    constraint_edges = set()
    edgesCDT_obstacles = []
    pts_hard_constraints = set()
    V2d_holes = []
    # add obstacles' edges
    for hole in holes:
        for seg in hole.segments:
            s, t = vertex_from_point[seg.start], vertex_from_point[seg.end]
            edge = []
            for n, pt in ((s, seg.start), (t, seg.end)):
                if pt not in pts_hard_constraints:
                    iCDT += 1
                    vertex_from_iCDT[iCDT + 3] = n
                    edge.append(iCDT)
                    pts_hard_constraints.add(pt)
                    V2d_holes.append(cdt.V2d(pt.x, pt.y))
                else:
                    edge.append(np.flatnonzero(vertex_from_iCDT == n)[0] - 3)
            st = (s, t) if s < t else (t, s)
            constraint_edges.add(st)
            edgesCDT_obstacles.append(cdt.Edge(*edge))

    # if adding obstacles, crossing-free edges might be removed from the mesh
    justly_removed = set()
    soft_constraints = set()
    if edgesCDT_obstacles:
        mesh.insert_vertices(V2d_holes)
        mesh.insert_edges(edgesCDT_obstacles)
        _, P_edges = planar_from_cdt_triangles(mesh,
                                               vertex_from_iCDT)
        # Here we use the changes in CDT triangulation to identify the P_A
        # edges that cross obstacles or lay in their vicinity.
        edges_to_examine = P_A_edges - P_edges
        edges_check = np.array(list(constraint_edges))
        while edges_to_examine:
            u, v = edges_to_examine.pop()
            uC, vC = VertexC[[u, v]]
            # if ⟨u, v⟩ does not cross any constraint_edges, add it to edgesCDT
            ovlap = find_edges_bbox_overlaps(VertexC, u, v, edges_check)
            if not any(is_crossing_no_bbox(uC, vC, *VertexC[edge])
                       for edge in edges_check[ovlap]):
                # ⟨u, v⟩ was removed from the triangulation but does not cross
                soft_constraints.add((u, v))
            else:
                # ⟨u, v⟩ crosses some constraint_edge
                justly_removed.add((u, v))
                # enlist for examination the up to 4 edges surrounding ⟨u, v⟩
                for s, t in ((u, v), (v, u)):
                    nb = P_A[s][t]['cw']
                    if nb == P_A[t][s]['cw']:
                        for p, q in ((nb, s) if nb < s else (s, nb),
                                     ((nb, t) if nb < t else (t, nb))):
                            if ((p, q) not in soft_constraints
                                and (p, q) not in justly_removed):
                                edges_to_examine.add((p, q))
        if soft_constraints:
            # add the crossing-free edges around obstacles as constraints
            edgesCDT_soft = [cdt.Edge(u if u >= 0 else T + R + u,
                                      v if v >= 0 else T + R + v)
                             for u, v in soft_constraints]
            mesh.insert_edges(edgesCDT_soft)

    # ######################################################################
    # E) Add concavities and get the Constrained Delaunay Triang.
    # ######################################################################
    debug('PART E')
    # create the PythonCDT edges
    edgesCDT_P_A = []

    # Add A's hull_concave as soft constraints to ensure A's edges remain in P.
    for s, t in zip(hull_concave, hull_concave[1:] + [hull_concave[0]]):
        s, t = (s, t) if s < t else (t, s)
        if ((s, t) in justly_removed
            or (s, t) in soft_constraints):
            # skip if ⟨s, t⟩ is known to cross an obstacle or was added earlier
            continue
        edgesCDT_P_A.append(cdt.Edge(s if s >= 0 else T + R + s,
                                     t if t >= 0 else T + R + t))
    mesh.insert_edges(edgesCDT_P_A)

    edgesCDT_concavities = []
    V2d_concavities = []
    # add concavities' edges
    for conc in concavities:
        for seg in conc.segments:
            s, t = vertex_from_point[seg.start], vertex_from_point[seg.end]
            edge = []
            for n, pt in ((s, seg.start), (t, seg.end)):
                if pt not in pts_hard_constraints:
                    iCDT += 1
                    vertex_from_iCDT[iCDT + 3] = n
                    edge.append(iCDT)
                    pts_hard_constraints.add(pt)
                    V2d_concavities.append(cdt.V2d(pt.x, pt.y))
                else:
                    edge.append(np.flatnonzero(vertex_from_iCDT == n)[0] - 3)
            st = (s, t) if s < t else (t, s)
            constraint_edges.add(st)
            edgesCDT_concavities.append(cdt.Edge(*edge))

    if edgesCDT_concavities:
        mesh.insert_vertices(V2d_concavities)
        mesh.insert_edges(edgesCDT_concavities)

    # ##########################################
    # Z) Scale coordinates back.
    # ##########################################
    # add any newly created plus the supertriangle's vertices to VertexC
    # note: B has already been increased by all stuntC lengths within the loop
    supertriangleC = (mean +
                      scale*np.array([(v.x, v.y) for v in mesh.vertices[:3]]))
    # NOTE: stuntC was scaled back upon its creation
    VertexC = np.vstack((VertexCʹ[:-R],
                         *stuntC,
                         supertriangleC,
                         VertexCʹ[-R:]))

    # Add length attribute to A's edges.
    A_edges = (*P_A_edges, *diagonals)
    source, target = zip(*A_edges)
    # TODO: ¿use d2roots for root-incident edges? probably not worth it
    A_edge_length = dict(
        zip(A_edges, (length.item() for length in
                      np.hypot(*(VertexC[source,] - VertexC[target,]).T))))
    nx.set_edge_attributes(A, A_edge_length, name='length')

    # ###############################################################
    # F) Build the planar embedding of the constrained triangulation.
    # ###############################################################
    debug('PART F')
    P_halfedge_pack, P_edges = planar_from_cdt_triangles(mesh,
                                                         vertex_from_iCDT)
    P = P_from_halfedge_pack(P_halfedge_pack)

    # Remove edges inside the concavities
    for hole in chain(
            concavityVertexSeqs,
            (([vertex_from_point[v] for v in hole.vertices[::-1]]
             for hole in holes) if obstacles else ())):
        for rev, cur, fwd in zip(chain((hole[-1],), hole[:-1]),
                                 hole, chain(hole[1:], (hole[0],))):
            while P[cur][fwd]['ccw'] != rev:
                u, v = cur, P[cur][fwd]['ccw']
                P.remove_edge(u, v)
                P_edges.remove((u, v) if u < v else (v, u))

    # adjust flat triangles around concavities
    #  changes_super = _flip_triangles_obstacles_super(
    #          P, T, B + 3, VertexC, max_tri_AR=max_tri_AR)

    convex_hull, to_remove, conc_outer_edges = hull_processor(
            P, T, supertriangle, vertex2conc_id_map)
    P.remove_edges_from(to_remove)
    P_edges.difference_update((u, v) if u < v else (v, u)
                              for u, v in to_remove)
    constraint_edges -= conc_outer_edges
    P.graph.update(R=R, T=T, B=B,
                   constraint_edges=constraint_edges,
                   supertriangleC=supertriangleC,)

    #  changes_obstacles = _flip_triangles_near_obstacles(P, T, B + 3,
    #                                                       VertexC)
    #  P.check_structure()
    #  print('changes_super', [(F[a], F[b]) for a, b in changes_super])
    #  print('changes_obstacles',
    #        [(F[a], F[b]) for a, b in changes_obstacles])

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
    P_edges.difference_update((u, v) for v in supertriangle for u in P[v])
    P_paths = nx.Graph(P_edges)

    # this adds diagonals to P_paths, but not diagonals that cross constraints
    for u, v in P_edges - hull_prunned_edges:
        # TODO: how is this check different from `if uv in constraint_edges`
        if (u in vertex2conc_id_map and v in vertex2conc_id_map
            and vertex2conc_id_map[u] == vertex2conc_id_map[v]):
            continue
        uvD = P[u][v]
        s, t = uvD['cw'], uvD['ccw']
        if is_triangle_pair_a_convex_quadrilateral(*VertexC[[u, v, s, t]]):
            P_paths.add_edge(s, t)

    nx.set_edge_attributes(P_paths, A_edge_length, name='length')
    #  for u, v in P_paths.edges - A_edge_length:
    for u, v, edgeD in P_paths.edges(data=True):
        if 'length' not in edgeD:
            edgeD['length'] = np.hypot(*(VertexC[u] - VertexC[v])).item()

    # #######################
    # X) Create hull_concave.
    # #######################
    # TODO: adjust comments, since this is done in section D.1
    #       (hull_concave is needed earlier)

    cw, ccw = rotation_checkers_factory(VertexC)

    A.graph['hull_concave'] = hull_concave

    # ###################################################################
    # H) Revisit A to update edges crossing borders with P_path contours.
    # ###################################################################
    debug('PART H')
    corner_to_A_edges = defaultdict(list)
    A_edges_to_revisit = []
    for u, v in A.edges - P_paths.edges:
        # For the edges in A that are not in P, we find their corresponding
        # shortest path in P_path and update the length attribute in A.
        length, path = nx.bidirectional_dijkstra(P_paths, u, v,
                                                 weight='length')
        debug('A_edge: %s–%s length: %.3f; path: %s', F[u], F[v], length, path)
        if all(n >= T for n in path[1:-1]):
            # keep only paths that only have border vertices between nodes
            edgeD = A[path[0]][path[-1]]
            midpath = (path[1:-1].copy()
                       if u < v else
                       path[-2:0:-1].copy())
            i = 0
            while i <= len(path) - 3:
                # Check if each vertex at the border is necessary.
                # The vertex is kept if the border angle and the path angle
                # point to the same side. Otherwise, remove the vertex.
                s, b, t = path[i:i + 3]
                # skip to shortcut if b is a neighbor of the supertriangle
                if all(n not in P[b] for n in supertriangle):
                    b_conc_id = vertex2conc_id_map[b]
                    debug('s: %s; b: %s; t: %s; b_conc_id: %s', F[s], F[b], F[t], b_conc_id)
                    debug([(F[n], vertex2conc_id_map.get(n)) for n in P.neighbors(b)])
                    nbs = P.neighbors_cw_order(b)
                    skip_test = True
                    for a in nbs:
                        if vertex2conc_id_map.get(a, -1) == b_conc_id:
                            skip_test = False
                            break
                    if skip_test:
                        i += 1
                        debug('Took the 1st continue.')
                        continue
                    skip_test = True
                    for c in nbs:
                        if (vertex2conc_id_map.get(c, -1) == b_conc_id
                                and c not in P[a]):
                            if P[b][a]['cw'] == c:
                                skip_test = False
                                break
                            a = c
                    if c == a:
                        # no nb remaining after making a = c, c <- first nb
                        c = next(P.neighbors_cw_order(b))
                        if P[b][a]['cw'] == c:
                            skip_test = False
                    debug('a: %d %s; c: %d %s; s: %d %s, t: %d %s; %s',
                         a, F[a], c, F[c], s, F[s], t, F[t], skip_test)
                    if (skip_test or not (cw(a, b, c)
                                          or ((a == s or cw(a, b, s))
                                              == cw(s, b, t)))):
                        i += 1
                        debug('Took the 2nd continue.')
                        continue
                # PERFORM SHORTCUT
                # TODO: The entire new path should go for a 2nd pass if it
                #       changed here. Unlikely to change in the 2nd pass.
                #       Reason: a shortcut may change the geometry in such
                #       way as to make additional shortcuts possible.
                del path[i + 1]
                length -= P_paths[s][b]['length'] + P_paths[b][t]['length']
                shortcut_length = np.hypot(*(VertexC[s] - VertexC[t]).T).item()
                length += shortcut_length
                # changing P_paths for the case of revisiting this block
                P_paths.add_edge(s, t, length=shortcut_length)
                shortcuts = edgeD.get('shortcuts')
                if shortcuts is None:
                    edgeD['shortcuts'] = [b]
                else:
                    shortcuts.append(b)
                debug('(%d) %s %s %s shortcut', i, F[s], F[b], F[t])
            edgeD.update(# midpath-> which P edges the A edge maps to
                         # (so that PathFinder works)
                         midpath=midpath,
                         # contour_... edges may include direct ones that are
                         # diverted because P_paths does not include them
                         kind='contour_'+edgeD['kind'])
            if len(path) > 2:
                edgeD['length'] = length
                u, v = (u, v) if u < v else (v, u)
                for p in path[1:-1]:
                    corner_to_A_edges[p].append((u, v))
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
    for uv in A_edges_to_revisit:
        st = diagonals.inv.get(uv)
        if st is not None:
            edgeD = A.edges[st]
            edgeD['kind'] = ('contour_delaunay'
                             if 'midpath' in edgeD else
                             'delaunay')
            del diagonals[st]
        # TODO: ¿how important is it to add ⟨s, t⟩ to P_A?
        # before removing ⟨u, v⟩, we should discern if it is usvt or utsv
        P_A.remove_edge(*uv)

    # ##################################################################
    # I) Revisit A to update d2roots according to lengths along P_paths.
    # ##################################################################
    debug('PART I')
    d2roots = cdist(VertexC[:T + B + 3], VertexC[-R:])
    # d2roots may not be the plain Euclidean distance if there are obstacles.
    if concavityVertexSeqs or obstacles:
        # Use P_paths to obtain estimates of d2roots taking into consideration
        # the concavities and obstacle zones.
        for r in range(-R, 0):
            lengths, paths = nx.single_source_dijkstra(P_paths, r,
                                                       weight='length')
            for n, path in paths.items():
                if n >= T or n < 0:
                    # skip border and root vertices
                    continue
                if any(p >= T for p in path):
                    # This estimate may be slightly longer that just going
                    # around the border.
                    debug('changing %d with path %s', n, path)
                    node_d2roots = A.nodes[n].get('d2roots')
                    if node_d2roots is None:
                        A.nodes[n]['d2roots'] = {r: d2roots[n, r]}
                    else:
                        node_d2roots.update({r: d2roots[n, r]})
                    d2roots[n, r] = lengths[n]

    # ##########################################
    # J) Calculate the area of the concave hull.
    # ##########################################
    if border is None:
        bX, bY = VertexC[convex_hull_A].T
    else:
        # for the bounding box, use border, roots and stunts
        bX, bY = np.vstack((VertexC[border], VertexC[-R:], *stuntC)).T
    # assuming that coordinates are UTM -> min() as bbox's offset to origin
    norm_offset = np.array((bX.min(), bY.min()), dtype=np.float64)
    # Take the sqrt() of the area and invert for the linear factor such that
    # area=1.
    norm_scale = 1./math.sqrt(
        area_from_polygon_vertices(*VertexC[hull_concave].T))

    # Set A's graph attributes.
    A.graph.update(
        T=T, R=R, B=B,
        VertexC=VertexC,
        border=border,
        name=L.name,
        handle=L.graph.get('handle', 'handleless'),
        planar=P_A,
        diagonals=diagonals,
        d2roots=d2roots,
        # TODO: make these 2 attribute names consistent across the code
        hull=convex_hull_A,
        hull_prunned=hull_prunned,
        hull_concave=hull_concave,
        # experimental attr
        norm_offset=norm_offset,
        norm_scale=norm_scale,
    )
    if obstacles:
        A.graph['obstacles'] = obstacles
    if stunts_primes:
        A.graph['num_stunts'] = len(stunts_primes)
    landscape_angle = L.graph.get('landscape_angle')
    if landscape_angle is not None:
        A.graph['landscape_angle'] = landscape_angle
    # products:
    # P: PlanarEmbedding
    # A: Graph (carries the updated VertexC)
    # P_A: PlanarEmbedding
    # P_paths: Graph
    # diagonals: dict
    return P, A


def delaunay(L: nx.Graph, bind2root: bool = False) -> nx.Graph:
    # TODO: deprecate the use of delaunay()
    _, A = make_planar_embedding(L)
    if bind2root:
        assign_root(A)
        R = L.graph['R']
        # assign each edge to the root closest to the edge's middle point
        VertexC = A.graph['VertexC']
        for u, v, edgeD in A.edges(data=True):
            edgeD['root'] = -R + np.argmin(
                    cdist(((VertexC[u] + VertexC[v])/2)[np.newaxis, :],
                          VertexC[-R:]))
    return A


def A_graph(G_base, delaunay_based=True, weightfun=None, weight_attr='weight'):
    # TODO: refactor to be compatible with interarray.mesh's delaunay()
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
    # A.remove_edges_from(list(A.edges(range(-R, 0))))
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
    R, T, B, D, VertexC, border, obstacles = (
        G.graph.get(k) for k in ('R', 'T', 'B', 'D', 'VertexC', 'border',
                                 'obstacles'))

    P = planar.copy()
    diagonals = A.graph['diagonals']
    P_A = A.graph['planar']
    seen_endpoints = set()
    for u, v in G.edges - P.edges:
        # update the planar embedding to include any Delaunay diagonals
        # used in G; the corresponding crossing Delaunay edge is removed
        u, v = (u, v) if u < v else (v, u)
        if u >= T:
            # we are in a redundant segment of a multi-segment path
            continue
        if v >= T and u not in seen_endpoints:
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
        G: nx.Graph, *, planar: nx.PlanarEmbedding, VertexC: np.ndarray,
        diagonals: bidict | None = None) -> nx.PlanarEmbedding:
    '''Ajust `planar` to include the edges actually used by reouteset `G`.

    Copies `planar` and flips the edges to their diagonal if the latter is an
    edge of `G`. Ideally, the returned PlanarEmbedding includes all `G` edges
    (an expected discrepancy are `G`'s gates).

    If `diagonals` is provided, some diagonal gates may become `planar`'s edges
    if they are not crossing any edge in `G`. Otherwise gates are ignored.

    Important: `G` must be free of edge×edge crossings.
    '''
    R, T, B, C, D = (G.graph.get(k, 0) for k in ('R', 'T', 'B', 'C', 'D'))
    border, obstacles, fnT = (
        G.graph.get(k) for k in ('border', 'obstacles', 'fnT'))
    if fnT is None:
        fnT = np.arange(R + T + B + 3 + C + D)
        fnT[-R:] = range(-R, 0)

    P = planar.copy()
    if diagonals is not None:
        diags = diagonals.copy()
    else:
        diags = ()
    debug('differences between G and P:')
    # get G's edges in terms of node range -R : T + B
    edges_G = {((u, v) if u < v else (v, u))
               for u, v in (fnT[edge,] for edge in G.edges)}
    ST = T + B
    edges_P = {((u, v) if u < v else (v, u))
               for u, v in P.edges if u < ST and v < ST}
    stack = list(edges_G - edges_P)
    # gates to the bottom of the stack
    stack.sort()
    while stack:
        u, v = stack.pop()
        if u < 0 and (u, v) not in diags:
            continue
        debug('%d–%d', u, v)
        intersection = set(planar[u]) & set(planar[v])
        if len(intersection) < 2:
            debug('share %d neighbors.', len(intersection))
            continue
        diagonal_found = False
        for s, t in combinations(intersection, 2):
            s, t = (s, t) if s < t else (t, s)
            if ((s, t) in edges_P
                and is_triangle_pair_a_convex_quadrilateral(
                    *VertexC[[s, t, u, v]])):
                diagonal_found = True
                break
        if not diagonal_found:
            if u >= 0:
                # only warn if the non-planar is not a gate
                warn('Failed to find flippable for non-planar %d–%d', u, v)
            continue
        if (s, t) in edges_G and u < 0:
            # not replacing edge with gate
            continue
        if planar[u][s]['ccw'] == t and planar[v][t]['ccw'] == s:
            # u-s-v-t already in ccw orientation
            pass
        elif planar[u][s]['cw'] == t and planar[v][t]['cw'] == s:
            # reassign so that u-s-v-t is in ccw orientation
            s, t = t, s
        else:
            debug('%d–%d–%d–%d is not in two triangles.', u, s, v, t)
            continue
        #  if not (s == planar[v][u]['ccw']
        #          and t == planar[v][u]['cw']):
        #      print(f'{F[u]}–{F[v]} is not in two triangles')
        #      continue
        #  if (s, t) not in planar:
        #      print(f'{F[s]}–{F[t]} is not in planar')
        #      continue
        debug('flipping %d–%d to %d–%d', s, t, u, v)
        P.remove_edge(s, t)
        if diags:
            # diagonal (u_, v_) is added to P -> forbid diagonals that cross it
            for (w, y) in ((u, s), (s, v), (v, t), (t, u)):
                wy = (w, y) if w < y else (y, w)
                diags.inv.pop(wy, None)
        P.add_half_edge(u, v, cw=s)
        P.add_half_edge(v, u, cw=t)
    return P
