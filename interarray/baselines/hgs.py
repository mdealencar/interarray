import math
from dataclasses import asdict
import numpy as np
import networkx as nx
import hygese as hgs
from py.io import StdCaptureFD

from ..interarraylib import fun_fingerprint
from ..repair import repair_routeset_path
from . import length_matrix_single_depot_from_G

# [[2012.10384] Hybrid Genetic Search for the CVRP: Open-Source Implementation
#  and SWAP\* Neighborhood]
# (https://arxiv.org/abs/2012.10384)

#  from ..utils import NodeTagger
#  F = NodeTagger()


def hgs_cvrp(A: nx.Graph, *, capacity: float, time_limit: float,
             vehicles: int | None = None, seed: int = 0) \
                     -> nx.Graph:
    '''Solves the OCVRP using PyHygese with links from `A`

    Wraps PyHygese, which provides bindings to the HGS-CVRP library (Hybrid
    Genetic Search solver for Capacitated Vehicle Routing Problems). This
    function uses it to solve an Open-CVRP i.e., vehicles do not return to the
    depot.

    Normalization of input graph is recommended before calling this function.

    Args:
        A: graph with allowed edges (if it has 0 edges, use complete graph)
        capacity: maximum vehicle capacity
        time_limit: [s] solver run time limit
        vehicles: number of vehicles (if None, use the minimum feasible)

    Returns:
        Solution topology.
    '''
    R, T, VertexC = (
        A.graph[k] for k in ('R', 'T', 'VertexC'))
    assert R == 1, 'ERROR: only single depot supported'

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
        seed=seed,
    )
    hgs_solver = hgs.Solver(parameters=ap, verbose=True)
    x_coordinates, y_coordinates = np.c_[VertexC[-R:].T, VertexC[:T].T]
    # data preparation
    # Distance_matrix may be provided instead of coordinates, or in addition to
    # coordinates. Distance_matrix is used for cost calculation if provided.
    # The additional coordinates will be helpful in speeding up the algorithm.
    demands = np.ones(T + R, dtype=float)
    demands[0] = 0.  # depot demand = 0
    weights, w_max = length_matrix_single_depot_from_G(A, scale=1.)
    vehicles_min = math.ceil(T/capacity)
    if vehicles is None or vehicles <= vehicles_min:
        if vehicles is not None and vehicles < vehicles_min:
            print(f'Vehicle number ({vehicles}) too low for feasibilty '
                  f'with capacity ({capacity}). Setting to {vehicles_min}.')
        # set to minimum feasible vehicle number
        vehicles = vehicles_min
    # HGS-CVRP crashes if distance_matrix has inf values, but there must
    # be a strong incentive to choose A edges only. (5× is arbitrary)
    distance_matrix = weights.clip(max=5*w_max)
    data = dict(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        distance_matrix=distance_matrix,
        service_times=np.zeros(T + R),
        demands=demands,
        vehicle_capacity=capacity,
        num_vehicles=vehicles,
        depot=0,
    )

    result, out, err = StdCaptureFD.call(hgs_solver.solve_cvrp, data,
                                         rounding=False)

    # create a topology graph S from the results
    S = nx.Graph(
        T=T, R=R,
        handle=A.graph['handle'],
        capacity=capacity,
        has_loads=True,
        objective=result.cost,
        creator='baselines.hgs',
        runtime=result.time,
        solver_log=out,
        solution_time=_solution_time(out, result.cost),
        method_options=dict(solver_name='HGS-CVRP',
                            complete=A.number_of_edges() == 0,
                            fun_fingerprint=fun_fingerprint()) | asdict(ap),
        #  solver_details=dict(
        #  )
    )
    branches = ([n - 1 for n in branch] for branch in result.routes)
    for subtree_id, branch in enumerate(branches):
        loads = range(len(branch), 0, -1)
        S.add_nodes_from(((n, {'load': load})
                          for n, load in zip(branch, loads)),
                         subtree=subtree_id)
        branch_roll = [-1] + branch[:-1]
        reverses = tuple(u < v for u, v in zip(branch, branch_roll))
        edgeD = ({'load': load, 'reverse': reverse}
                 for load, reverse in zip(loads, reverses))
        S.add_edges_from(zip(branch_roll, branch, edgeD))
    root_load = sum(S.nodes[n]['load'] for n in S.neighbors(-1))
    S.nodes[-1]['load'] = root_load
    assert root_load == T, 'ERROR: root node load does not match T.'
    return S


def _solution_time(log, objective) -> float:
    sol_repr = f'{objective:.2f}'
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


def iterative_hgs_cvrp(A: nx.Graph, *, capacity: float, time_limit: float,
                       vehicles: int | None = None, seed: int = 0,
                       max_iter: int = 10) -> nx.Graph:
    '''Iterate until crossing-free solution is found (`hgs_cvrp()` wrapper).

    Each time a solution with a crossing is produced, one of the offending
    edges is removed from `A` and the solver is called again. In the same
    way as `hgs_cvrp()`, it is recommended to pass a normalized `A`.

    Args:
        *: see `hgs_cvrp()`
        max_iter: maximum number of `hgs_cvrp()` calls in serie

    Returns:
        Solution S
    '''

    def remove_solve_repair(edge, Aʹ, num_crossings):
        # TODO: use a filtered subgraph view instead of copying
        A = Aʹ.copy()
        A.remove_edge(*edge)
        S = hgs_cvrp(A, capacity=capacity, time_limit=time_limit,
                     vehicles=vehicles, seed=seed)
        S = repair_routeset_path(S, A)
        return S, A, (S.graph.get('num_crossings', 0) < num_crossings)

    # solve
    S = hgs_cvrp(A, capacity=capacity, time_limit=time_limit,
                 vehicles=vehicles, seed=seed)
    # repair
    S = repair_routeset_path(S, A)
    # TODO: accumulate solution_time throughout the iterations
    #       (makes sense to add a new field)
    for i in range(max_iter):
        crossings = S.graph.get('outstanding_crossings', [])
        if not crossings:
            break
        # there are still crossings
        crossing_resolved = False
        for edge in crossings[0]:
            # try removing one edge at a time from A
            S, Aʹ, succeeded = remove_solve_repair(edge, A, len(crossings))
            if succeeded:
                # TODO: maybe try comparing the quality between the edge removals
                A = Aʹ
                crossing_resolved = True
                break
        if not crossing_resolved:
            print('WARNING: Failed to resolve crossing! Will keep trying.')
            # use the A with the last edge removed
            A = Aʹ
    if i > 0:
        S.graph['hgs_reruns'] = i
        if i == 9:
            print('Probably got stuck in an infinite loop')
    return S
