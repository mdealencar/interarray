import math
import time
import os
import re
import io
import tempfile
import subprocess
from pathlib import Path
import networkx as nx
import numpy as np

from .utils import length_matrix_single_depot_from_G
from ..interarraylib import fun_fingerprint


# TODO: Deprecate that. Unable to make LKH work in ACVRP with EDGE_FILE
#       It seems like EDGE_FILE is for candidate edges of the transformed
#       (symmetric) problem (i.e., LKH expects a higher node count)
def make_edge_listing(A: nx.Graph, scale: float) -> str:
    '''
    LKH's EDGE_FILE param (follows Concorde format)
    node numbering starts from 0
    every node number will be incremented by 1 when loaded by LKH
    '''
    R = A.graph['R']
    T = A.graph['T']
    V = R + T
    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    return '\n'.join((
        # first line is <number_of_nodes> <number_of_edges>
        f'{V} {2*A_nodes.number_of_edges() + 2*T}',
        # edges not including the depot (symmetric)
        '\n'.join('{s} {t} {cost:.0f}\n{t} {s} {cost:.0f}'.format(
            s=u+1, t=v+1, cost=scale*d)
                  for u, v, d in A_nodes.edges(data='length')),
        # depot connects to all nodes, but asymmetrically (zero-cost return)
        '\n'.join(f'{1} {n} {scale*d:.0f}\n{n} 1 0'
                  for n, d in enumerate(A.graph['d2roots'][:, 0], start=1)),
        ))


def lkh_acvrp(A: nx.Graph, *, capacity: int, time_limit: int,
              scale: float = 1e4, vehicles: int | None = None,
              runs: int = 50, per_run_limit: float = 15.) -> nx.Graph:
    '''
    Lin-Kernighan-Helsgaun via LKH-3 binary.
    Asymmetric Capacitated Vehicle Routing Problem.

    Arguments:
        `A`: graph with allowed edges (if it has 0 edges, use complete graph)
        `capacity`: maximum vehicle capacity
        `time_limit`: [s] solver run time limit
        `scale`: factor to scale lengths (should be < 1e6)
        `vehicles`: number of vehicles (if None, use the minimum feasible)
        `runs`: consult LKH manual
        `per_run_limit`: consult LKH manual
    '''
    R, T, B, VertexC = (
        A.graph.get(k) for k in ('R', 'T', 'B', 'VertexC'))
    assert R == 1, 'LKH allows only 1 depot'
    problem_fname = 'problem.txt'
    params_fname = 'params.txt'
    edge_fname = 'edge_file.txt'
    w_saturation = np.iinfo(np.int32).max/1000
    d2roots = A.graph.get('d2roots')
    if d2roots is None:
        d2roots = cdist(VertexC[:-R], VertexC[-R:])
        A.graph['d2roots'] = d2roots
    weights, w_max = length_matrix_single_depot_from_G(A, scale=scale)
    assert w_max <= w_saturation, 'ERROR: weight values outside int32 range.'
    weights = weights.clip(max=w_saturation).round().astype(np.int32)
    if w_max > w_saturation:
        print('WARNING: at least one edge weight has been clipped.')
    #  weights = np.ones((T + R, T + R), dtype=int)
    with io.StringIO() as str_io:
        np.savetxt(str_io, weights, fmt='%d')
        edge_weights = str_io.getvalue()[:-1]

    #  print(edge_weights)
    output_fname = 'solution.out'
    specs = dict(
        NAME=A.graph.get('name', 'unnamed'),
        TYPE='ACVRP',  # maybe try asymmetric TSP: 'ATSP',
        # TYPE='ATSP',  # maybe try asymmetric capacitaded VRP: 'ACVRP',
        DIMENSION=T + R,  # CVRP number of nodes and depots
        # DIMENSION=T,  # TSP: number of nodes
        CAPACITY=capacity,
        EDGE_WEIGHT_TYPE='EXPLICIT',
        EDGE_WEIGHT_FORMAT='FULL_MATRIX',
    )
    data = dict(
        # DEMAND_SECTION='\n'.join(f'{i} 1' for i in range(T)) + f'\n{T} 0',
        DEPOT_SECTION=f'1\n-1',
        EDGE_WEIGHT_SECTION=edge_weights,
    )

    if False and A is not None:
        # TODO: Deprecate this. Passing edges always makes LKH fail.
        specs['EDGE_DATA_FORMAT'] = 'ADJ_LIST'
        A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
        data['EDGE_DATA_SECTION'] = '\n'.join((
            # depot has out-edges to all nodes
            f'1 {" ".join(str(n) for n in range(2, T + 2))} -1',
            # all nodes have out-edges to depot
            '\n'.join(f'{n + 2} 1 {" ".join(str(a + 2) for a in adj)} -1'
                      for n, adj in A_nodes.adjacency()),
            '-1'
            ))
    if False and A is not None:
        # TODO: Deprecate this. Passing edges always makes LKH fail.
        specs['EDGE_DATA_FORMAT'] = 'EDGE_LIST'
        data['EDGE_DATA_SECTION'] = '\n'.join((
                '\n'.join(f'{u + 2} {v + 2}\n{v + 2} {u + 2}'
                          for u, v in A.edges if u >= 0 and v >= 0),
                '\n'.join(f'{n} 1\n1 {n}' for n in range(2, T + 2)),
                '-1\n'))
    if False and A is not None:
        # TODO: Deprecate this. EDGE_FILE is for the transformed problem.
        edge_str = make_edge_listing(A, scale)

    spec_str = '\n'.join(f'{k}: {v}' for k, v in specs.items())
    data_str = '\n'.join(f'{k}\n{v}' for k, v in data.items()) + '\nEOF'
    vehicles_min = math.ceil(T/capacity)
    if (vehicles is None) or (vehicles <= vehicles_min):
        # set to minimum feasible vehicle number
        if vehicles is not None and vehicles < vehicles_min:
            print(f'Vehicle number ({vehicles}) too low for feasibilty '
                  f'with capacity ({capacity}). Setting to {vehicles_min}.')
        vehicles = vehicles_min
        min_route_size = (T % capacity) or capacity
    else:
        min_route_size = 0
    params = dict(
        SPECIAL=None,  # None -> output only the key
        #  PRECISION=1000,  # d[i][j] = PRECISION*c[i][j] + pi[i] + pi[j]
        PRECISION=100,  # d[i][j] = PRECISION*c[i][j] + pi[i] + pi[j]
        TOTAL_TIME_LIMIT=time_limit,
        TIME_LIMIT=per_run_limit,
        RUNS=runs,  # default: 10
        # MAX_TRIALS=100,  # default: number of nodes (DIMENSION)
        # TRACE_LEVEL=1,  # default is 1, 0 supresses output
        #  INITIAL_TOUR_ALGORITHM='GREEDY',  # { … | CVRP | MTSP | SOP } Default: WALK
        VEHICLES=vehicles,
        MTSP_MIN_SIZE=min_route_size,
        MTSP_MAX_SIZE=capacity,
        MTSP_OBJECTIVE='MINSUM',  # [ MINMAX | MINMAX_SIZE | MINSUM ]
        MTSP_SOLUTION_FILE=output_fname,
        #  MOVE_TYPE='5 SPECIAL',  # <integer> [ SPECIAL ]
        #  GAIN23='NO',
        #  KICKS=1,
        #  KICK_TYPE=4,
        #  MAX_SWAPS=0,
        #  POPULATION_SIZE=12,  # default 10
        #  PATCHING_A=
        #  PATCHING_C=
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        problem_fpath = os.path.join(tmpdir, problem_fname)
        params['PROBLEM_FILE'] = problem_fpath
        params_fpath = os.path.join(tmpdir, params_fname)
        params['MTSP_SOLUTION_FILE'] = os.path.join(tmpdir,
                                                    output_fname)
        if False and A is not None:
            edge_fpath = os.path.join(tmpdir, edge_fname)
            params['EDGE_FILE'] = edge_fpath
            Path(edge_fpath).write_text(edge_str)
        Path(problem_fpath).write_text('\n'.join((spec_str, data_str)))
        Path(params_fpath).write_text('\n'.join(
            (f'{k} = {v}' if v is not None else k) for k, v in params.items()))
        start_time = time.perf_counter()
        result = subprocess.run(['LKH', params_fpath], capture_output=True)
        elapsed_time = time.perf_counter() - start_time
        output_fpath = os.path.join(tmpdir, output_fname)
        if Path(output_fpath).is_file():
            with open(output_fpath, 'r') as f_sol:
                penalty, minimum = next(f_sol).split(':')[-1][:-1].split('_')
                next(f_sol)  # discard second line
                branches = [[int(node) - 2 for node in line.split(' ')[1:-5]]
                            for line in f_sol]
        else:
            penalty = 0
            minimum = 'inf'
            branches = []
    log = result.stdout.decode('utf8')
    S = nx.Graph(
        creator='baselines.lkh',
        T=T, R=R,
        handle=A.graph['handle'],
        has_loads=True,
        capacity=capacity,
        objective=float(minimum)/scale,
        method_options=dict(
            solver_name='LKH-3',
            complete=A.number_of_edges() == 0,
            scale=scale,
            type=specs['TYPE'],
            time_limit=time_limit,
            runs=runs,
            per_run_limit=per_run_limit,
            fun_fingerprint=fun_fingerprint()),
        runtime=elapsed_time,
        solver_log=log,
        solution_time=_solution_time(log, minimum),
        penalty=int(penalty),
        #  solver_details=dict(
        #  )
    )
    if not penalty or result.stderr:
        print('===stdout===', result.stdout.decode('utf8'), sep='\n')
        print('===stderr===', result.stderr.decode('utf8'), sep='\n')
    else:
        #  tail = result.stdout[result.stdout.rfind(b'Successes/'):].decode('ascii')
        tail = result.stdout[result.stdout.rfind(b'Successes/'):].decode()
        entries = iter(tail.splitlines())
        # Decision to drop avg. stats: unreliable, possibly due to time_limit
        next(entries)  # skip sucesses line
        S.graph['cost_extrema'] = tuple(float(v) for v in re.match(
            r'Cost\.min = (-?\d+), Cost\.avg = -?\d+\.?\d*,'
            r' Cost\.max = -?(\d+)',
            next(entries)).groups())
        next(entries)  # skip gap line
        S.graph['penalty_extrema'] = tuple(float(v) for v in re.match(
            r'Penalty\.min = (\d+), Penalty\.avg = \d+\.?\d*,'
            r' Penalty\.max = (\d+)',
            next(entries)).groups())
        S.graph['trials_extrema'] = tuple(float(v) for v in re.match(
            r'Trials\.min = (\d+), Trials\.avg = \d+\.?\d*,'
            r' Trials\.max = (\d+)',
            next(entries)).groups())
        S.graph['runtime_extrema'] = tuple(float(v) for v in re.match(
            r'Time\.min = (\d+\.?\d*) sec., Time\.avg = \d+\.?\d* sec.,'
            r' Time\.max = (\d+\.?\d*) sec.',
            next(entries)).groups())
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
    sol_repr = f'{objective}'
    time = 0.
    for line in log.splitlines():
        if not line or line[0] == '*':
            continue
        if line[:4] == 'Run ':
            # example: Run 4: Cost = 84_129583, Time = 2.87 sec.
            cost_, time_ = line.split(': ')[1].split(', ')
            # example time_: Time = 2.87 sec.
            time += float(time_.split(' = ')[1].split(' ')[0])
            # example cost_: Cost = 84_8724588,
            if cost_.split('_')[1] == sol_repr:
                return time
