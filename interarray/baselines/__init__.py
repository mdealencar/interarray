from typing import Tuple
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

from ..geometric import delaunay


def weight_matrix_single_depot_from_G(G: nx.Graph, *, precision_factor: float)\
        -> Tuple[np.ndarray, nx.Graph]:
    """

    Edge cost matrix for VRP-based solvers.
    It is assumed that the problem has been pre-scaled, such that multiplying
    all weights by `precision_factor` will place them within the int range.
    """
    M = G.graph['M']
    assert M == 1, 'ERROR: only single depot supported'
    VertexC = G.graph['VertexC']
    N = VertexC.shape[0] - M
    d2roots = G.graph.get('d2roots')
    if d2roots is None:
        d2roots = cdist(VertexC[:-M], VertexC[-M:])
        G.graph['d2roots'] = d2roots
    # using max int32 value (2_147_483_647), because this is used in the
    # context of old C libraries
    weight = np.full((N + M, N + M), 2_147_483_647, dtype=int)
    A = delaunay(G)
    for u, v, w in A.edges(data='length'):
        weight[u + 1, v + 1] = weight[v + 1, u + 1] = round(w*precision_factor)
    weight[0, 1:] = np.round(d2roots[:, 0]*precision_factor).astype(int)
    # make return to depot always free
    weight[:, 0] = 0
    return weight, A
