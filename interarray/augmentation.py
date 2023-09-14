# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import itertools

import numpy as np
from matplotlib.path import Path


def poisson_disc_filler(N, min_dist, boundary, exclude=None, seed=None,
                        iter_max_factor=50, partial_fulfilment=True):
    '''
    Fills the area delimited by `boundary` with `N` randomly
    placed points that are at least `min_dist` apart and that
    don't fall inside any of the `exclude` areas.
    :param N:
    :param min_dist:
    :param boundary: iterable (B × 2) with CCW-ordered vertices of a polygon
                     special case: if B == 2 then boundary is a rectangle and
                     the two vertices represent (min_x, min_y) and (max_x,
                     max_y)
    :param exclude: iterable (E × X × 2)
    :param iter_max_factor: factor to multiply by `N` to limit the number of
                            iterations
    :param partial_fulfilment: whether to return less than `N` points (True) or
                               to raise exception (False) if unable to fulfill
                               request.
    :return numpy array shaped (N, 2) with points' positions
    '''
    # [Poisson-Disc Sampling](https://www.jasondavies.com/poisson-disc/)

    if len(boundary) > 2:
        path = Path(boundary)
        bX, bY = boundary.T
        lower_bound = np.array((bX.min(), bY.min()), dtype=float)
        upper_bound = np.array((bX.max(), bY.max()), dtype=float)
        area_avail = 0.5*np.abs(np.dot(bX, np.roll(bY, 1))
                                - np.dot(bY, np.roll(bX, 1)))
    else:
        path = False
        lower_bound, upper_bound = np.array(boundary)
        area_avail = np.prod((upper_bound - lower_bound) + min_dist)

    # TODO: implement exclusion zones
    if exclude is not None:
        raise NotImplementedError

    # quick check for outrageous densities
    # circle packing efficiency limit: η = π srqt(3)/6 = 0.9069
    # A Simple Proof of Thue's Theorem on Circle Packing
    # https://arxiv.org/abs/1009.4322
    area_demand = min_dist**2*np.pi*N/4
    if (not partial_fulfilment
            and (area_demand > np.pi*np.sqrt(3)/6*area_avail)):
        raise ValueError("(N, min_dist) given are beyond the ideal circle "
                         "packing for the boundary area.")
    # friendly warning for high densities (likely to place less than N points)
    if 2*area_demand > area_avail:
        print('WARNING: Unlikely to fulfill with current arguments - '
              'try a lower density.')

    iter_max = iter_max_factor*N
    threshold = min_dist**2
    rng = np.random.default_rng(seed=seed)

    I, J = np.ogrid[-2:3, -2:3]
    # mask for the 20 neighbors
    # (5x5 grid excluding corners and center)
    neighbormask = ((abs(I) != 2) | (abs(J) != 2)) & ((I != 0) | (J != 0))

    # create auxiliary grid covering the defined boundary
    cell_size = min_dist/np.sqrt(2)
    i_len, j_len = np.ceil(((upper_bound - lower_bound)/cell_size)).astype(int)

    # grid for mapping of cell to point (initialized as NaNs)
    cells = np.full((i_len, j_len, 2), np.nan, dtype=float)

    def no_overlap(p: int, q: int, candidateC: np.ndarray) -> bool:
        '''
        Check for overlap over the 20 cells neighboring the current cell.
        :param p:  x cell index.
        :param q:  y cell index.
        :param candidateC: numpy array shaped (2,) with the point's coordinates
        :return True if point does not overlap, False otherwise.
        '''
        P = I + p
        Q = J + q
        mask = (neighbormask
                & (0 <= P) & (P <= (i_len - 1))
                & (0 <= Q) & (Q <= (j_len - 1)))
        i_, j_ = np.nonzero(mask)
        neighbors = cells[P[i_, 0], Q[0, j_]]
        for neighborC in neighbors[~np.isnan(neighbors[:, 0])]:
            if sum((candidateC - neighborC)**2) < threshold:
                return False
        return True

    # list of indices of empty cells.
    empty_cell_idc = list(itertools.product(range(i_len), range(j_len)))

    dart = np.empty((2,), dtype=float)
    pos = np.empty((N, 2), dtype=float)
    out_count = 0
    for iter_count in range(1, iter_max + 1):
        # pick random empty cell
        empty_idx = rng.integers(low=0, high=len(empty_cell_idc))
        i, j = empty_cell_idc[empty_idx]

        # dart throw inside cell
        dart[:] = (i + rng.uniform(), j + rng.uniform())
        dart *= cell_size
        dart += lower_bound

        # check boundary and overlap
        if (((not path and (dart <= upper_bound).all())
                or (path and path.contains_point(dart)))
                and no_overlap(i, j, dart)):
            # add new point and remove cell from empty list
            pos[out_count] = dart
            cells[empty_cell_idc[empty_idx]] = dart
            del empty_cell_idc[empty_idx]
            out_count += 1
            if out_count == N or not empty_cell_idc:
                break

    if out_count < N:
        pos = pos[:out_count]
        msg = (f'Only {out_count} points generated (requested: {N}, '
               f'iterations: {iter_count}).')
        if partial_fulfilment:
            print('WARNING:', msg)
        else:
            raise ValueError(msg)
    return pos
