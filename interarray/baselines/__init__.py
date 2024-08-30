import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


def length_matrix_single_depot_from_G(G: nx.Graph, scale: float)\
        -> tuple[np.ndarray, float]:
    """Edge length matrix for VRP-based solvers.
    It is assumed that the problem has been pre-scaled, such that multiplying
    all lengths by `scale` will place them within a numerically stable range.
    Length of return to depot from all nodes is set to 0 (i.e. Open-VRP).
    Order of nodes in the returned matrix is depot, clients (required by some
    VRP methods), which differs from interrarray order (i.e clients, depot).

    Parameters
    ----------
    G: NetworkX Graph
        Must contain graph attributes `M` and `VertexC`. If G has no edges,
        calculate lengths of the complete graph of G's nodes, otherwise only
        calculate lengths of G's edges and assign +inf to non-existing edges.
        G's edges must have the 'length' attribute.

    scale: float
        Factor to multiply all lengths by.

    Returns
    -------
    L, len_max:
        Matrix of lengths and maximum length value (below +inf).
    """
    M, N, VertexC = (G.graph.get(k) for k in ('M', 'N', 'VertexC'))
    assert M == 1, 'ERROR: only single depot supported'
    if G.number_of_edges() == 0:
        # bring depot to before the clients
        VertexCmod = np.r_[VertexC[-M:], VertexC[:N]]
        L = cdist(VertexCmod, VertexCmod)*scale
        len_max = L.max()
    else:
        d2roots = G.graph.get('d2roots')
        if d2roots is None:
            d2roots = cdist(VertexC[:-M], VertexC[-M:])
            G.graph['d2roots'] = d2roots
        # non-available edges will have infinite length
        L = np.full((N + M, N + M), np.inf)
        len_max = d2roots[:, 0].max()
        for u, v, length in G.edges(data='length'):
            L[u + 1, v + 1] = L[v + 1, u + 1] = length*scale
            len_max = max(len_max, length)
        L[0, 1:] = d2roots[:N, 0]*scale
        len_max *= scale
    # make return to depot always free
    L[:, 0] = 0.
    return L, len_max
