import math
import time
from dataclasses import asdict
import numpy as np
import networkx as nx
import hygese as hgs

from ..interarraylib import calcload
from ..pathfinding import PathFinder
from ..geometric import make_graph_metrics
from . import weight_matrix_single_depot_from_G

# [[2012.10384] Hybrid Genetic Search for the CVRP: Open-Source Implementation
#  and SWAP\* Neighborhood]
# (https://arxiv.org/abs/2012.10384)


def pyhygese(G_base: nx.Graph, *, capacity: float, time_limit: int,
             precision_factor=1e8) -> nx.Graph:
    G = nx.create_empty_copy(G_base)
    M = G.graph['M']
    assert M == 1, 'ERROR: only single depot supported'
    # Solver initialization
    # https://github.com/vidalt/HGS-CVRP/tree/main#running-the-algorithm
    # class AlgorithmParameters:
    #     nbGranular: int = 20  # Granular search parameter, limits the number of moves in the RI local search
    #     mu: int = 25  # Minimum population size
    #     lambda_: int = 40  # Number of solutions created before reaching the maximum population size (i.e., generation size).
    #     nbElite: int = 4  # Number of elite individuals
    #     nbClose: int = 5  # Number of closest solutions/individuals considered when calculating diversity contribution
    #     targetFeasible: float = 0.2  # target ratio of feasible individuals between penalty updates
    #     seed: int = 0  # fixed seed
    #     nbIter: int = 20000  # max iterations without improvement
    #     timeLimit: float = 0.0  # seconds
    #     useSwapStar: bool = True

    ap = hgs.AlgorithmParameters(
        timeLimit=time_limit,  # seconds
        # nbIter=2000,  # max iterations without improvement (20,000)
    )
    start_time = time.perf_counter()
    hgs_solver = hgs.Solver(parameters=ap, verbose=True)
    run_time = time.perf_counter() - start_time
    VertexC = G.graph['VertexC']
    N = VertexC.shape[0] - M
    VertexCmod = np.r_[VertexC[-M:], VertexC[:N]]
    # data preparation
    data = dict()
    data['x_coordinates'], data['y_coordinates'] = VertexCmod.T

    weight, A = weight_matrix_single_depot_from_G(
            G, precision_factor=precision_factor)

    # Distance_matrix may be provided instead of coordinates, or in addition to
    # coordinates. Distance_matrix is used for cost calculation if provided.
    # The additional coordinates will be helpful in speeding up the algorithm.
    data['distance_matrix'] = weight

    data['service_times'] = np.zeros(N + M)
    demands = np.ones(N + M)
    demands[0] = 0  # depot demand = 0
    data['demands'] = demands
    data['vehicle_capacity'] = capacity
    data['num_vehicles'] = math.ceil(N/capacity)
    data['depot'] = 0

    result = hgs_solver.solve_cvrp(data)

    for branch in result.routes:
        s = -1
        for t in branch:
            t -= 1
            G.add_edge(s, t, length=np.hypot(*(VertexC[s] - VertexC[t]).T))
            s = t

    calcload(G)
    PathFinder(G).create_detours(in_place=True)
    make_graph_metrics(G)
    # G.graph['iterations'] = ???
    G.graph['capacity'] = capacity
    # G.graph['overfed'] = [len(G[root])/np.ceil(N/capacity)*M
    #                       for root in roots]
    G.graph['edges_created_by'] = 'PyHygese'
    G.graph['edges_fun'] = pyhygese
    G.graph['creation_options'] = asdict(ap)
    G.graph['runtime_unit'] = 's'
    G.graph['runtime'] = run_time
    return G
