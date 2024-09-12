# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from .utils import NodeTagger

import shapely as shp
import numpy as np
import numba as nb
import networkx as nx

F = NodeTagger()


# iCDF_factory(N_min = 70,  N_max = 200, η = 0.6, d_lb = 0.045):
def iCDF_factory(N_min: int, N_max: int, η: float, d_lb: float)\
        -> Callable[[float], int]:
    '''
    Create function to shape the PDF: y(x) = d(N) - d_lb = 2*sqrt(η/π/N) - d_lb
    where:
        N is the WT count
        η is the area covered by N circles of diameter d (η = Nπd²/4)
        d_lb is the lower bound for the minimum distance between WT
    '''

    def integral(x):  # integral of y(x) wrt x
        return 4*np.sqrt(x*η/np.pi) - d_lb*x

    def integral_inv(y):  # integral_inv(integral(x)) = x
        return ((-4*np.sqrt(4*η**2 - np.pi*η*d_lb*y) + 8*η - np.pi*d_lb*y)
                / (np.pi*d_lb**2))

    offset = integral(N_min - 0.4999999)
    area_under_curve = integral(N_max + 0.5) - offset

    def iCDF(u: float) -> int:
        '''Map from u ~ uniform(0, 1) to random variable N ~ custom \
        probability density function'''
        return int(round(integral_inv(u*area_under_curve + offset)))

    return iCDF


def normalize_site_single_oss(G: nx.Graph)\
        -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    DEPRECATED: use interarraylib's `normalized()` and `as_single_oss()`

    Calculate the area and scale the boundary so that it has area 1.
    The boundary and OSS are translated to the 1st quadrant, near the origin.

    IF SITE HAS MULTIPLE OSSs, ONLY 1 IS RETURNED (mean of the OSSs' coords).
    '''
    VertexC = G.graph['VertexC']
    BoundaryC = VertexC[G.graph['border']]
    bound_poly = shp.Polygon(BoundaryC)
    corner_lo, corner_hi = tuple(np.array(bound_poly.bounds[A:B])
                                 for A, B in ((0, 2), (2, 4)))
    M = G.graph['M']
    factor = 1/np.sqrt(bound_poly.area)
    BoundaryC -= corner_lo
    BoundaryC *= factor
    oss = ((VertexC[-M:].mean(axis=0) - corner_lo)*factor)[np.newaxis, :]
    return factor, corner_lo, BoundaryC, oss, (corner_hi - corner_lo)*factor


def build_instance_graph(WTpos, boundary, name='', handle='unnamed', oss=None,
                         landscape_angle=0):
    N = WTpos.shape[0]
    if oss is not None:
        M = oss.shape[0]
        VertexC = np.concatenate((WTpos, oss))
    else:
        M = 0
        VertexC = WTpos
    G = nx.Graph(
        name=name,
        handle=handle,
        M=M,
        boundary=boundary,
        landscape_angle=landscape_angle,
        VertexC=VertexC)
    G.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'})
                      for r in range(-M, 0)))
    return G


@nb.njit(cache=True, inline='always')
def clears(repellers: nb.float64[:, :], clearance_sq: np.float64,
           point: nb.float64[:]) -> bool:
    '''
    Evaluate if `point` (2,) is more that sqrt(clearance_sq) away from all
    `repellers` (K×2).
    Return boolean.
    '''
    return (((point[np.newaxis, :] - repellers)**2).sum(axis=1)
            >= clearance_sq).all()


def contains_np(polygon: nb.float64[:, :],
                pts: nb.float64[:, :]) -> nb.bool_[:]:
    '''
    Evaluate if 2D points in `pts` (N×2) are inside `polygon` (K×2, CCW vertex
    order).
    Return 1D boolean array (True if pts[i] inside `polygon`).
    '''
    polygon_rolled = np.roll(polygon, -1, axis=0)
    vectors = polygon_rolled - polygon
    mask1 = (pts[:, None] == polygon).all(-1).any(-1)
    m1 = ((polygon[:, 1] > pts[:, None, 1])
          != (polygon_rolled[:, 1] > pts[:, None, 1]))
    slope = ((pts[:, None, 0] - polygon[:, 0]) * vectors[:, 1]) - (
            vectors[:, 0] * (pts[:, None, 1] - polygon[:, 1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (polygon_rolled[:, 1] < polygon[:, 1])
    m4 = m1 & m3
    count = np.count_nonzero(m4, axis=-1)
    mask3 = ~(count % 2 == 0)
    return mask1 | mask2 | mask3


@nb.njit(cache=True)
def contains(polygon: nb.float64[:, :], point: nb.float64[:]) -> bool:
    '''
    Evaluate if `point` (2,) is inside `polygon` (K×2, CCW vertex
    order).
    Return boolean.
    '''
    intersections = 0
    dx2, dy2 = point - polygon[-1]

    for p in polygon:
        dx, dy = dx2, dy2
        dx2, dy2 = point - p

        F = (dx - dx2)*dy - dx*(dy - dy2)
        if np.isclose(F, 0., rtol=0.) and dx*dx2 <= 0 and dy*dy2 <= 0:
            return True

        if (dy >= 0 and dy2 < 0) or (dy2 >= 0 and dy < 0):
            if F > 0:
                intersections += 1
            elif F < 0:
                intersections -= 1
    return intersections != 0


def poisson_disc_filler(N: int, min_dist: float, boundary: nb.float64[:, :],
                        repellers: nb.optional(nb.float64[:, :]) = None,
                        clearance: float = 0, exclude=None, seed=None,
                        iter_max_factor: int = 50, plot: bool = False,
                        partial_fulfilment: bool = True) -> nb.float64[:, :]:
    '''
    Fills the area delimited by `boundary` with `N` randomly
    placed points that are at least `min_dist` apart and that
    don't fall inside any of the `repellers` discs or `exclude` areas.
    :param N:
    :param min_dist:
    :param boundary: iterable (B × 2) with CCW-ordered vertices of a polygon
    :param repellers: iterable (M × 2) with the centers of forbidden discs
    :param clearance: the radius of the forbidden discs
    :param exclude: iterable (E × X × 2)
    :param iter_max_factor: factor to multiply by `N` to limit the number of
                            iterations
    :param partial_fulfilment: whether to return less than `N` points (True) or
                               to raise exception (False) if unable to fulfill
                               request.
    :return numpy array shaped (N, 2) with points' positions
    '''
    # TODO: implement exclusion zones
    if exclude is not None:
        raise NotImplementedError

    bound_poly = shp.Polygon(boundary)
    area_avail = bound_poly.area
    corner_lo, corner_hi = tuple(np.array(bound_poly.bounds[A:B])
                                 for A, B in ((0, 2), (2, 4)))

    # quick check for outrageous densities
    # circle packing efficiency limit: η = π srqt(3)/6 = 0.9069
    # A Simple Proof of Thue's Theorem on Circle Packing
    # https://arxiv.org/abs/1009.4322
    area_demand = N*np.pi*min_dist**2/4
    efficiency = area_demand/area_avail
    efficiency_optimal = np.pi*np.sqrt(3)/6
    if efficiency > efficiency_optimal:
        msg = (f"(N = {N}, min_dist = {min_dist}) imply a packing "
               f"efficiency of {efficiency:.3f} which is higher than "
               f"the optimal possible ({efficiency_optimal:.3f}).")
        if partial_fulfilment:
            print('Info: Attempting partial fullfillment.', msg,
                  'Try with lower N and/or min_dist.')
        else:
            raise ValueError(msg)

    # create auxiliary grid covering the defined boundary
    cell_size = min_dist/np.sqrt(2)
    i_len, j_len = np.ceil(
        (corner_hi - corner_lo)/cell_size
    ).astype(np.int_)
    boundary_scaled = (boundary - corner_lo)/cell_size
    if repellers is None:
        repellers_scaled = None
        clearance_sq = 0.
    else:
        repellers_scaled = (repellers - corner_lo)/cell_size
        clearance_sq = (clearance/cell_size)**2

    # Alternate implementation using np.mgrid
    #  pts = np.reshape(
    #      np.moveaxis(np.mgrid[0: i_len + 1, 0: j_len + 1], 0, -1),
    #      ((i_len + 1)*(j_len + 1), 2)
    #  )
    pts = np.empty(((i_len + 1)*(j_len + 1), 2), dtype=int)
    pts_temp = pts.reshape((i_len + 1, j_len + 1, 2))
    pts_temp[..., 0] = np.arange(i_len + 1)[:, np.newaxis]
    pts_temp[..., 1] = np.arange(j_len + 1)[np.newaxis, :]
    inside = contains_np(boundary_scaled, pts).reshape((i_len + 1, j_len + 1))

    # reduce 2×2 sub-matrices of `inside` with logical_or (i.e. .any())
    cell_covers_polygon = np.lib.stride_tricks.as_strided(
        inside, shape=(2, 2, inside.shape[0] - 1, inside.shape[1] - 1),
        strides=inside.strides*2, writeable=False
    ).any(axis=(0, 1))

    # check boundary's vertices
    for k, (i, j) in enumerate(boundary_scaled.astype(int)):
        if not cell_covers_polygon[i, j]:
            ij = boundary_scaled[k].copy()
            direction = boundary_scaled[k - 1] - ij
            direction /= np.linalg.norm(direction)
            to_mark = [(i, j)]
            while True:
                nbr = (cell_covers_polygon[max(0, i - 1), j]
                       or cell_covers_polygon[min(i_len - 1, i + 1), j]
                       or cell_covers_polygon[i, max(0, j - 1)]
                       or cell_covers_polygon[i, min(j_len - 1, j + 1)])
                if nbr:
                    break
                ij += direction*0.999
                i, j = ij.astype(int)
                to_mark.append((i, j))
            for i, j in to_mark:
                cell_covers_polygon[i, j] = True

    # Sequence of (i, j) of cells that overlap with the polygon
    cell_idc = np.argwhere(cell_covers_polygon)

    iter_max = iter_max_factor*N
    rng = np.random.default_rng(seed)

    # useful plot for debugging purposes only
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(cell_covers_polygon.T, origin='lower',
                  extent=[0, cell_covers_polygon.shape[0],
                          0, cell_covers_polygon.shape[1]])
        ax.scatter(*np.nonzero(inside), marker='.')
        ax.scatter(*boundary_scaled.T, marker='x')
        ax.plot(*np.vstack((boundary_scaled, boundary_scaled[:1])).T)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.grid()

    # point-placing function
    points = wrapped_poisson_disc_filler(
        N, iter_max, i_len, j_len, cell_idc, boundary_scaled, clearance_sq,
        repellers_scaled, rng)

    # check if request was fulfilled
    if len(points) < N:
        msg = (f'Only {len(points)} points generated (requested: {N}, itera'
               f'tions: {iter_max}, efficiency requested: {efficiency:.3f}, '
               f'efficiency limit: {efficiency_optimal:.3f})')
        print('WARNING:', msg)

    return points*cell_size + corner_lo


@nb.njit(cache=True)
def wrapped_poisson_disc_filler(
        N: int, iter_max: int, i_len: int, j_len: int,
        cell_idc: nb.int64[:, :], boundary_scaled: nb.float64[:, :],
        clearance_sq: float, repellers_scaled: nb.optional(nb.float64[:, :]),
        rng: np.random.Generator) -> nb.float64[:, :]:
    '''See `poisson_disc_filler()`.'''
    # [Poisson-Disc Sampling](https://www.jasondavies.com/poisson-disc/)

    # mask for the 20 neighbors
    # (5x5 grid excluding corners and center)
    neighbormask = np.array(((False, True, True,  True, False),
                             (True,  True, True,  True, True),
                             (True,  True, False, True, True),
                             (True,  True, True,  True, True),
                             (False, True, True,  True, False)))

    # points to be returned by this function
    points = np.empty((N, 2), dtype=np.float64)
    # grid for mapping of cell to position in array `points` (N means not set)
    cells = np.full((i_len, j_len), N, dtype=np.int64)

    def no_conflict(p: int, q: int, point: nb.float64[:]) -> bool:
        '''
        Check for conflict with points from the 20 cells neighboring the
        current cell.
        :param p:  x cell index.
        :param q:  y cell index.
        :param point: numpy array shaped (2,) with the point's coordinates
        :return True if point does not conflict, False otherwise.
        '''
        p_min, p_max = max(0, p - 2), min(i_len, p + 3)
        q_min, q_max = max(0, q - 2), min(j_len, q + 3)
        cells_window = cells[p_min:p_max, q_min:q_max].copy()
        mask = (neighbormask[2 + p_min - p: 2 + p_max - p,
                             2 + q_min - q: 2 + q_max - q]
                & (cells_window < N))
        ii = cells_window.reshape(mask.size)[np.flatnonzero(mask.flat)]
        return not (((point[np.newaxis, :] - points[ii])**2).sum(axis=-1)
                    < 2).any()

    out_count = 0
    idc_list = list(range(len(cell_idc)))

    # dart-throwing loop
    for iter_count in range(1, iter_max + 1):
        # pick random empty cell
        empty_idx = rng.integers(low=0, high=len(idc_list))
        ij = cell_idc[idc_list[empty_idx]]
        i, j = ij

        # dart throw inside cell
        dartC = ij + rng.random(2)

        # check boundary, overlap and clearance
        if contains(boundary_scaled, dartC):
            if no_conflict(i, j, dartC):
                if repellers_scaled is not None:
                    if not clears(repellers_scaled, clearance_sq, dartC):
                        continue
                # add new point and remove cell from empty list
                points[out_count] = dartC
                cells[i, j] = out_count
                del idc_list[empty_idx]
                out_count += 1
                if out_count == N or not idc_list:
                    break

    return points[:out_count]
