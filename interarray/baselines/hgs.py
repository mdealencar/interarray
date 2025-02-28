import math
from typing import Sequence
import numpy as np
import networkx as nx
from multiprocessing import Pool

from ..interarraylib import fun_fingerprint
#  from ..interarraylib import NodeTagger
from ..repair import repair_routeset_path
from ..clustering import clusterize
from .utils import length_matrix_single_depot_from_G
from ._core_hgs import do_hgs
#  from .. import warn, debug, info

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

    solver_options = dict(
        timeLimit=time_limit,  # seconds
        # nbIter=2000,  # max iterations without improvement (20,000)
        seed=seed,
    )
    coordinates = np.c_[VertexC[-R:].T, VertexC[:T].T]
    # data preparation
    # Distance_matrix may be provided instead of coordinates, or in addition to
    # coordinates. Distance_matrix is used for cost calculation if provided.
    # The additional coordinates will be helpful in speeding up the algorithm.
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

    routes, runtime, solution_time, cost, log, algo_params = do_hgs(
        distance_matrix, coordinates, vehicles, capacity, solver_options)

    # create a topology graph S from the results
    S = nx.Graph(
        T=T, R=R,
        handle=A.graph['handle'],
        capacity=capacity,
        has_loads=True,
        objective=cost,
        creator='baselines.hgs',
        runtime=runtime,
        solver_log=log,
        solution_time=solution_time,
        method_options=dict(solver_name='HGS-CVRP',
                            complete=A.number_of_edges() == 0,
                            fun_fingerprint=fun_fingerprint()) | algo_params,
        #  solver_details=dict(
        #  )
    )
    branches = ([n - 1 for n in branch] for branch in routes)
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


def _length_matrices(
        A: nx.Graph, cluster_: list[set[int]],
        num_slack_: Sequence[int],
        clip_factor: float = 5.) -> tuple[list, list]:
    d2roots = A.graph['d2roots']
    R = A.graph['R']
    W_ = []
    indices_ = []
    w_max = 0.
    for r, (cluster, num_slack) in enumerate(zip(cluster_, num_slack_), start=-R):
        n_from_i = np.array([r] + sorted(cluster) + [r]*num_slack, dtype=int)
        terminal_slice = slice(1, -num_slack if num_slack else None)
        i_from_n = {n: i for i, n in enumerate(n_from_i[terminal_slice], 1)}
        # non-available edges will have infinite length
        A_clu = nx.subgraph_view(A, filter_node=lambda n: n in cluster)
        W_dim = len(cluster) + num_slack + 1
        W_clu = np.full((W_dim, W_dim), np.inf)
        for u, v, length in A_clu.edges(data='length'):
            idx = i_from_n[u], i_from_n[v]
            # terminal-terminal distances are symmetric
            W_clu[idx] = W_clu[idx[::-1]] = length
            w_max = max(w_max, length)

        # depot distances are asymmetric
        # fill the distances from depot
        W_clu[0, terminal_slice] = d2roots[n_from_i[terminal_slice], r]
        # make return to depot always free
        W_clu[:, 0] = 0.

        if num_slack:
            # make the slack nodes only connect to all terminals and from depot
            # from slack to each terminal (same as depot to each terminal)
            W_clu[-num_slack:, terminal_slice] = W_clu[0, terminal_slice]
            # from depot to slack (free)
            W_clu[0, -num_slack:] = 0.

        W_.append(W_clu)
        indices_.append(n_from_i)
    # only after preparing all matrices we have the actual w_max
    for W in W_:
        np.clip(W, a_min=None, a_max=clip_factor*w_max, out=W)
    return W_, indices_


def hgs_multiroot(A: nx.Graph, *, capacity: int, time_limit: float,
                  balanced: bool = False, seed: int = 0) -> nx.Graph:
    R, T = (A.graph[k] for k in 'RT')
    VertexC = A.graph['VertexC']

    # Partition location in clusters and get link lengths from A
    cluster_, num_slack_ = clusterize(A, capacity)
    W_, indices_ = _length_matrices(A, cluster_,
        num_slack_ if balanced else [0]*len(cluster_))

    # HGS-CVRP parameters
    solver_options = dict(
        timeLimit=time_limit,  # seconds
        # nbIter=2000,  # max iterations without improvement (20,000)
        seed=seed,
    )

    # data preparation
    # Distance_matrix may be provided instead of coordinates, or in addition to
    # coordinates. Distance_matrix is used for cost calculation if provided.
    # The additional coordinates will be helpful in speeding up the algorithm.
    cluster_data = zip(
        W_,  # distance matrix
        [VertexC[indices].T for indices in indices_],  # coordinates
        [math.ceil(len(cluster)/capacity) for cluster in cluster_], # vehicles
        [capacity]*R,  # vehicle capacity
        [solver_options]*R,  # to be **passed to `hgs.AlgorithmParameters()`
    )

    # Launch one parallel HGS-CVRP solver process per root.
    # TODO: do not assume there are more CPU cores than roots
    pool = Pool(R)
    results = pool.starmap(do_hgs, cluster_data)
    routes_, runtime_, solution_time_, cost_, log_, algo_params = zip(*results)

    #  print([[[F[i] for i in indices[route]] for route in routes] for routes, indices in zip (routes_, indices_)])
    if balanced:
        # remove the slack nodes from the routes
        for num_slack, routes, cluster in zip(num_slack_, routes_, cluster_):
            if num_slack != 0:
                num_nodes = len(cluster) + 1
                routes[:] = [[n for n in route if n < num_nodes]
                             for route in routes]
    # create a topology graph S from the results
    S = nx.Graph(
        T=T, R=R,
        handle=A.graph['handle'],
        capacity=capacity,
        has_loads=True,
        objective=sum(cost_),
        creator='baselines.hgs',
        runtime=max(runtime_),
        solver_log=log_,
        solution_time=solution_time_,
        method_options=dict(solver_name='HGS-CVRP',
                            complete=A.number_of_edges() == 0,
                            fun_fingerprint=fun_fingerprint()) | algo_params[0],
        #  solver_details=dict(
        #  )
    )
    subtree_id_start = 0
    for r, (routes, indices) in enumerate(zip(routes_, indices_), start=-R):
        branches = (indices[route] for route in routes)
        for subtree_id, branch in enumerate(branches, start=subtree_id_start):
            loads = range(len(branch), 0, -1)
            S.add_nodes_from(((n, {'load': load})
                              for n, load in zip(branch, loads)),
                             subtree=subtree_id)
            branch_roll = np.empty_like(branch)
            branch_roll[1:] = branch[:-1]
            branch_roll[0] = r
            reverses = tuple(u < v for u, v in
                             zip(branch, branch_roll))
            edgeD = ({'load': load, 'reverse': reverse}
                     for load, reverse in zip(loads, reverses))
            S.add_edges_from(zip(branch_roll.tolist(),
                                 branch.tolist(), edgeD))
        subtree_id_start = subtree_id + 1
        root_load = sum(S.nodes[n]['load'] for n in S.neighbors(r))
        S.nodes[r]['load'] = root_load
    assert sum(S.nodes[r]['load'] for r in range(-R, 0)) == T, 'ERROR: root node load does not match T.'
    return S
