# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import inspect
import itertools
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]


class NodeStr():

    def __init__(self, fnT, N):
        self.fnT = fnT
        self.N = N

    def __call__(self, u, *args):
        nodes = tuple((self.fnT[n], n)
                      for n in (u,) + args if n is not None)
        out = '–'.join(F[n_] + ('' if n < self.N else f'({F[n]})')
                       for n_, n in nodes)
        if len(nodes) > 1:
            out = f'«{out}»'
        else:
            out = f'<{out}>'
        return out


class NodeTagger():
    # 50 digits, 'I' and 'l' were dropped
    alphabet = 'abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ'
    value = {c: i for i, c in enumerate(alphabet)}

    def __getattr__(self, b50):
        dec = 0
        digit_value = 1
        if b50[0] < 'α':
            for digit in b50[::-1]:
                dec += self.value[digit]*digit_value
                digit_value *= 50
            return dec
        else:
            # for greek letters, only single digit is implemented
            return ord('α') - ord(b50[0]) - 1

    def __getitem__(self, dec):
        if dec is None:
            return '∅'
        elif isinstance(dec, str):
            return dec
        b50 = []
        if dec >= 0:
            while True:
                dec, digit = divmod(dec, 50)
                b50.append(self.alphabet[digit])
                if dec == 0:
                    break
            return ''.join(b50[::-1])
        else:
            return chr(ord('α') - dec - 1)


F = NodeTagger()


class Alerter():

    def __init__(self, where, varname):
        self.where = where
        self.varname = varname
        self.f_creation = inspect.stack()[1].frame

    def __call__(self, text):
        i = self.f_creation.f_locals[self.varname]
        function = inspect.stack()[1].function
        if self.where(i, function):
            print(f'[{i}|{function}] ' + text)


def G_from_TG(T, G_base, capacity=None, load_col=4, weight_is_length=False):
    '''Creates a networkx graph with nodes and data from G_base and edges from
    a T matrix.
    T matrix: [ [u, v, length, load (WT number), cable type], ...]'''
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    M = G_base.graph['M']
    N = G_base.number_of_nodes() - M

    # indexing differences:
    # T starts at 1, while G starts at 0
    # T begins with OSSs followed by WTGs,
    # while G begins with WTGs followed by OSSs
    # the line bellow converts the indexing:
    edges = (T[:, :2].astype(int) - M - 1) % (N + M)

    G.add_weighted_edges_from(zip(*edges.T, T[:, 2]), weight='length')
    # nx.set_edge_attributes(G, {(u, v): load for (u, v), load
                               # in zip(edges, T[:, load_col])},
                           # name='load')
    # try:
    calcload(G)
    # except AssertionError as err:
        # print(f'>>>>>>>> SOMETHING WENT REALLY WRONG: {err} <<<<<<<<<<<')
        # return G
    if T.shape[1] >= 4:
        for (u, v), load in zip(edges, T[:, load_col]):
            Gload = G.edges[u, v]['load']
            assert Gload == load, (
                f'<G.edges[{u}, {v}]> {Gload} != {load} <T matrix>')
    if weight_is_length:
        for u, v, nodeD in G.edges(data=True):
            nodeD['weight'] = nodeD['length']
    G.graph['has_loads'] = True
    G.graph['edges_created_by'] = 'G_from_TG()'
    G.graph['prevented_crossings'] = 0
    if capacity is not None:
        G.graph['overfed'] = [len(G[root])/np.ceil(N/capacity)*M
                              for root in range(N, N + M)]
    return G


def cost(G):
    if 'cables' not in G.graph:
        return np.inf
    cables = G.graph['cables']
    total = 0.
    N_cables = len(cables.cost)
    for u, v, data in G.edges(data=True):
        cable_idx = cables.capacity.searchsorted(data['load'])
        if cable_idx == N_cables:
            # available cables do not meet required capacity
            return np.inf
        total += data['length']*cables.cost[cable_idx]
    return total


def new_graph_like(G_base, edges=None):
    '''copies graph and nodes attributes, but not edges'''
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    if edges:
        G.add_edges_from(edges)
    return G


def deprecated_make_graph_metrics(G):
    '''This function changes G in place!
    Calculates for all nodes, for each root node:
    - distance to root nodes
    - angle wrt root node
    '''
    VertexC = G.graph['VertexC']
    M = G.graph['M']
    # N = G.number_of_nodes() - M
    roots = range(-M, 0)
    NodeC = VertexC[:-M]
    RootC = VertexC[-M:]

    # calculate distance from all nodes to each of the roots
    d2roots = np.hstack(tuple(cdist(rootC[np.newaxis, :], NodeC).T
                              for rootC in RootC))

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
    # G.graph['anglesRank'] = np.argsort(np.argsort(angles, axis=0), axis=0)

    # BEGIN dark magic
    # checks if two consecutive node angles are within a tolerance (atol)
    # (consecutive in their sorted angle values, not in the sequence they
    # appear in the angles array)
    argsorted = np.argsort(angles, axis=0)
    twiceargsorted = np.argsort(argsorted, axis=0)
    prevanglesI = np.take_along_axis(
        argsorted,
        (twiceargsorted - 1) % len(angles),
        axis=0)
    prevangles = np.take_along_axis(angles, prevanglesI, axis=0)
    # wrap-around of angles must be treated separetely
    # output = []
    for pa, firstI in zip(prevangles.T, argsorted[0]):
        pa[firstI] -= 2*np.pi
        # output.append(F[firstI])
    # print(output)

    keep_rank = np.isclose(angles, prevangles, atol=0.5/180.*np.pi)

    for i, firstI in enumerate(argsorted[0]):
        if keep_rank[firstI, i]:
            # TODO: do that
            print('>>>>> TODO: implement angle wraping with same rank <<<<<')

    # TODO: change that line
    G.graph['anglesRank'] = np.take_along_axis(
        np.cumsum(np.take_along_axis(~keep_rank, argsorted, axis=0),
                  axis=0),
        twiceargsorted,
        axis=0)
    # for i, (arsor, ang, kr, ar) in enumerate(zip(argsorted.T, angles.T,
                                                 # keep_rank.T,
                                                 # G.graph['anglesRank'].T)):
        # print(f'root {i}')
        # for j, line in enumerate(list(zip(ang/np.pi*180, kr, ar))):
            # print(F[j], line)
    # END: dark magic

    anglesMag = abs(angles)
    G.graph['anglesYhp'] = angles >= 0.
    G.graph['anglesXhp'] = anglesMag < np.pi/2


def make_graph_metrics(G):
    '''This function changes G in place!
    Calculates for all nodes, for each root node:
    - distance to root nodes
    - angle wrt root node
    '''
    VertexC = G.graph['VertexC']
    M = G.graph['M']
    # N = G.number_of_nodes() - M
    roots = range(-M, 0)
    NodeC = VertexC[:-M]
    RootC = VertexC[-M:]

    # calculate distance from all nodes to each of the roots
    d2roots = np.hstack(tuple(cdist(rootC[np.newaxis, :], NodeC).T
                              for rootC in RootC))

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


def update_lengths(G):
    '''Adds missing edge lengths.
    Changes G in place.'''
    VertexC = G.graph['VertexC']
    for u, v, dataE in G.edges.data():
        if 'length' not in dataE:
            dataE['length'] = np.hypot(*(VertexC[u] - VertexC[v]).T)


def calcload(G):
    '''calculates the number of nodes on the leaves side of each edge
    this function will update the edges' "load" property of graph G'''
    M = G.graph['M']
    N = G.number_of_nodes() - M
    D = G.graph.get('D')
    if D is not None:
        N -= D
    roots = range(-M, 0)
    for node, data in G.nodes(data=True):
        if 'load' in data:
            del data['load']

    def count_descendants(parent, children, subtree):
        '''recurse down the tree, returning total descendant nodes'''
        nodeD = G.nodes[parent]
        if not children:
            nodeD['load'] = 1
            return 1
        default = 0 if nodeD['type'] == 'detour' else 1
        load = nodeD.get('load', default)
        for child in children:
            G.nodes[child]['subtree'] = subtree
            grandchildren = set(G[child].keys())
            grandchildren.remove(parent)
            childload = count_descendants(child, grandchildren, subtree)
            G[parent][child].update((
                ('load', childload),
                ('reverse', parent > child)
            ))
            load += childload
        nodeD['load'] = load
        return load

    counted = 0
    subtree = 0
    for root in roots:
        for subroot in G[root]:
            previous = G.nodes[root].get('load', 1)
            counted += count_descendants(root, [subroot], subtree) - previous
            subtree += 1
    assert counted == N, f'counted ({counted}) != nonrootnodes({N})'
    G.graph['has_loads'] = True


def pathdist(G, path):
    'measures the length (distance) of a `path` of nodes in `G`'
    VertexC = G.graph['VertexC']
    dist = 0.
    p = path[0]
    for n in path[1:]:
        dist += np.hypot(*(VertexC[p] - VertexC[n]).T)
        p = n
    return dist

def G_from_site(site):
    VertexC = site['VertexC']
    M = site['M']
    N = len(VertexC) - M
    G = nx.Graph(name=site.get('name', ''),
                 M=M,
                 VertexC=VertexC,
                 boundary=site['boundary'])

    G.add_nodes_from(((n, {'label': F[n], 'type': 'wtg'})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'type': 'oss'})
                      for r in range(-M, 0)))
    return G

def poisson_disc_filler(N, min_dist, boundary, exclude=None,
                        iter_max_factor=30, seed=None, partial_fulfilment=True):
    '''
    Fills the area delimited by `boundary` with `N` randomly
    placed points that are at least `min_dist` apart and that
    don't fall inside any of the `exclude` areas.
    :param N:
    :param min_dist:
    :param boundary: iterable (B × 2) with CCW-ordered vertices of a polygon
                     special case: if B == 2 then boundary is a rectangle and the
                     two vertices represent (min_x, min_y) and (max_x, max_y)
    :param exclude: iterable (E × X × 2)
    :param iter_max_factor: factor to multiply by `N` to limit the number of iterations
    :param partial_fulfilment: whether to return less than `N` points (True) or to
                               raise exception (False) if unable to fulfill request.
    :return numpy array shaped (N, 2) with points' positions
    '''
    # [Poisson-Disc Sampling](https://www.jasondavies.com/poisson-disc/)

    # TODO: implement polygonal areas
    if len(boundary) > 2:
        raise NotImplementedError
    else:
        lower_bound, upper_bound = np.array(boundary)

    # TODO: implement exclusion zones
    if exclude is not None:
        raise NotImplementedError


    # quick check for outrageous densities
    # circle packing efficiency limit: η = π srqt(3)/6 = 0.9069
    # A Simple Proof of Thue's Theorem on Circle Packing
    # https://arxiv.org/abs/1009.4322
    area_avail = np.prod((upper_bound - lower_bound) + min_dist)
    area_demand = min_dist**2*np.pi*N/4
    if not partial_fulfilment and (area_demand > np.pi*np.sqrt(3)/6*area_avail):
        raise ValueError("(N, min_dist) given are beyond the ideal circle packing for the boundary area.")
    # friendly warning for high densities (likely to place less than N points)
    if 2*area_demand > area_avail:
        print('WARNING: Unlikely to fulfill with current arguments - try a lower density.')

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

    def no_overlap(p:int, q:int, candidateC: np.ndarray) -> bool:
        '''
        Check for overlap over the 20 cells neighboring the current cell.
        :param p:  x cell index.
        :param q:  y cell index.
        :param candidateC: numpy array shaped (2,) with the point's coordinates.
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
        if (dart <= upper_bound).all() and no_overlap(i, j, dart):
            # add new point and remove cell from empty list
            pos[out_count] = dart
            cells[empty_cell_idc[empty_idx]] = dart
            del empty_cell_idc[empty_idx]
            out_count += 1
            if out_count == N or not empty_cell_idc:
                break

    if out_count < N:
        pos = pos[:out_count]
        msg = f'Only {out_count} points generated (requested: {N}, iterations: {iter_count}).'
        if partial_fulfilment:
            print('WARNING:', msg)
        else:
            raise ValueError(msg)
    return pos
