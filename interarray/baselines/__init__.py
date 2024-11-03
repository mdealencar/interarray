import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


def length_matrix_single_depot_from_G(
        A: nx.Graph, *, scale: float
        ) -> tuple[np.ndarray, float]:
    """Edge length matrix for VRP-based solvers.
    It is assumed that the problem has been pre-scaled, such that multiplying
    all lengths by `scale` will place them within a numerically stable range.
    Length of return to depot from all nodes is set to 0 (i.e. Open-VRP).
    Order of nodes in the returned matrix is depot, clients (required by some
    VRP methods), which differs from interrarray order (i.e clients, depot).

    Parameters
    ----------
    A: NetworkX Graph
        Must contain graph attributes `R`, 'T', `VertexC` and 'd2roots'. If A
        has no edges, calculate lengths of the complete graph of A's nodes,
        otherwise only calculate lengths of A's edges and assign +inf to
        non-existing edges. A's edges must have the 'length' attribute.

    scale: float
        Factor to multiply all lengths by.

    Returns
    -------
    L, len_max:
        Matrix of lengths and maximum length value (below +inf).
    """
    R, T, VertexC, d2roots = (A.graph.get(k)
                              for k in ('R', 'T', 'VertexC', 'd2roots'))
    assert R == 1, 'ERROR: only single depot supported'
    if A.number_of_edges() == 0:
        # bring depot to before the clients
        VertexCmod = np.r_[VertexC[-R:], VertexC[:T]]
        L = cdist(VertexCmod, VertexCmod)*scale
        len_max = L.max()
    else:
        # non-available edges will have infinite length
        L = np.full((T + R, T + R), np.inf)
        len_max = d2roots[:T, 0].max()
        for u, v, length in A.edges(data='length'):
            L[u + 1, v + 1] = L[v + 1, u + 1] = length*scale
            len_max = max(len_max, length)
        L[0, 1:] = d2roots[:T, 0]*scale
        len_max *= scale
    # make return to depot always free
    L[:, 0] = 0.
    return L, len_max
