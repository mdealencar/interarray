import math
from dataclasses import asdict
import numpy as np
import networkx as nx
import hygese as hgs
from py.io import StdCaptureFD

from ..interarraylib import calcload, fun_fingerprint
from ..pathfinding import PathFinder
from ..geometric import make_graph_metrics
from . import length_matrix_single_depot_from_G

# [[2012.10384] Hybrid Genetic Search for the CVRP: Open-Source Implementation
#  and SWAP\* Neighborhood]
# (https://arxiv.org/abs/2012.10384)

#  from ..utils import NodeTagger
#  F = NodeTagger()


def hgs_cvrp(G_base: nx.Graph, *, capacity: float, time_limit: int,
             A: nx.Graph | None, scale: float = 1e4,
             vehicles: int | None = None) -> nx.Graph:
    '''Wrapper for PyHygese module, which provides bindings to the HGS-CVRP
    library (Hybrid Genetic Search solver for Capacitated Vehicle Routing
    Problems).

    Normalization of input graph is recommended before calling this function.

    Arguments:
        `G_base`: graph with the site's coordinates and boundary
        `capacity`: maximum vehicle capacity
        `time_limit`: [s] solver run time limit
        `A`: graph with allowed edges (if None, use complete graph)
        `scale`: factor to scale lengths
        `vehicles`: number of vehicles (if None, use the minimum feasible)
    '''
    G = nx.create_empty_copy(G_base)
    M, N, B, VertexC, border, exclusions, landscape_angle = (
        G.graph.get(k) for k in ('M', 'N', 'B', 'VertexC', 'border',
                                 'exclusions', 'landscape_angle'))
    assert M == 1, 'ERROR: only single depot supported'
    # Solver initialization
    # https://github.com/vidalt/HGS-CVRP/tree/main#running-the-algorithm
    # class AlgorithmParameters:
    #     nbGranular: int = 20  # Granular search parameter, limits the number
    #                           # of moves in the RI local search
    #     mu: int = 25  # Minimum population size
    #     lambda_: int = 40  # Number of solutions created before reaching the
    #                        # maximum population size (i.e., generation size).
    #     nbElite: int = 4  # Number of elite individuals
    #     nbClose: int = 5  # Number of closest solutions/individuals consider-
    #                       # ed when calculating diversity contribution
    #     targetFeasible: float = 0.2  # target ratio of feasible individuals
    #                                  # between penalty updates
    #     seed: int = 0  # fixed seed
    #     nbIter: int = 20000  # max iterations without improvement
    #     timeLimit: float = 0.0  # seconds
    #     useSwapStar: bool = True

    ap = hgs.AlgorithmParameters(
        timeLimit=time_limit,  # seconds
        # nbIter=2000,  # max iterations without improvement (20,000)
    )
    hgs_solver = hgs.Solver(parameters=ap, verbose=True)
    VertexCmod = np.c_[VertexC[-M:].T, VertexC[:N].T]*scale
    # data preparation
    # Distance_matrix may be provided instead of coordinates, or in addition to
    # coordinates. Distance_matrix is used for cost calculation if provided.
    # The additional coordinates will be helpful in speeding up the algorithm.
    demands = np.ones(N + M, dtype=float)
    demands[0] = 0.  # depot demand = 0
    weights, w_max = length_matrix_single_depot_from_G(A or G, scale)
    if A is not None:
        d2roots = A.graph.get('d2roots')
    else:
        d2roots = G.graph.get('d2roots')
    if d2roots is None:
        d2roots = weights[0, 1:, None].copy()/scale
        G.graph['d2roots'] = d2roots
    vehicles_min = math.ceil(N/capacity)
    if vehicles is None or vehicles <= vehicles_min:
        if vehicles is not None and vehicles < vehicles_min:
            print(f'Vehicle number ({vehicles}) too low for feasibilty '
                  f'with capacity ({capacity}). Setting to {vehicles_min}.')
        # set to minimum feasible vehicle number
        vehicles = vehicles_min
    if A is not None:
        # HGS-CVRP crashes if distance_matrix has inf values, but there must
        # be a strong incentive to choose A edges only. (5× is arbitrary)
        distance_matrix = weights.clip(max=5*w_max)
    else:
        distance_matrix = weights
    data = dict(
        x_coordinates=VertexCmod[0],
        y_coordinates=VertexCmod[1],
        distance_matrix=distance_matrix,
        service_times=np.zeros(N + M),
        demands=demands,
        vehicle_capacity=capacity,
        num_vehicles=vehicles,
        depot=0,
    )

    result, out, err = StdCaptureFD.call(hgs_solver.solve_cvrp, data)

    nonAedges = []
    for branch in result.routes:
        s = -1
        t = branch[0] - 1
        G.add_edge(s, t, length=G.graph['d2roots'][t, s])
        #  print(f'{F[s]}–{F[t]}', end='')
        s = t
        for t in branch[1:]:
            t -= 1
            #  print(f'–{F[t]}', end='')
            if A is not None and (s, t) in A.edges:
                G.add_edge(s, t, length=A[s][t]['length'])
            else:
                #  print(f'WARNING: edge {F[s]}-{F[t]} is not in A')
                G.add_edge(s, t, length=np.hypot(*(VertexC[s] - VertexC[t]).T))
                nonAedges.append((s, t))
            s = t

    calcload(G)
    if nonAedges:
        G.graph['nonAedges'] = nonAedges
    else:
        pass
        #  PathFinder(G).create_detours(in_place=True)
    G.graph.update(
        capacity=capacity,
        undetoured_length=result.cost/scale,
        edges_created_by='PyHygese',
        edges_fun=hgs_cvrp,
        creation_options=dict(complete=A is None,
                              scale=scale) | asdict(ap),
        runtime_unit='s',
        runtime=result.time,
        solver_log=out,
        solution_time=_solution_time(out, result.cost),
        fun_fingerprint=fun_fingerprint(),
    )
    return G


def _solution_time(log, undetoured_length) -> float:
    sol_repr = f'{undetoured_length:.2f}'
    for line in log.splitlines():
        if line[0] == '-':
            continue
        fields = line.split(' | ')
        if fields[2] != 'NO-FEASIBLE':
            incumbent = fields[2].split(' ')[2]
        else:
            incumbent = ''
        if incumbent == sol_repr:
            _, time = fields[1].split(' ')
            return float(time)
    # if sol_repr was not found, return total runtime
    return float(line.split(' ')[-1])


def get_sol_time(G: nx.Graph) -> float:
    """Graph must have graph attribute 'solver_log'"""
    log = G.graph['solver_log']
    sol = G.graph['undetoured_length']*G.graph['creation_options']['scale']
    return _solution_time(log, sol)
