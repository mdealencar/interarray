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

from . import length_matrix_single_depot_from_G
from ..interarraylib import calcload, fun_fingerprint
from ..pathfinding import PathFinder
from ..geometric import make_graph_metrics, delaunay


# TODO: Deprecate that. Unable to make LKH work in ACVRP with EDGE_FILE
#       It seems like EDGE_FILE is for candidate edges of the transformed
#       (symmetric) problem (i.e., LKH expects a higher node count)
def make_edge_listing(A: nx.Graph, scale: float) -> str:
    '''
    LKH's EDGE_FILE param (follows Concorde format)
    node numbering starts from 0
    every node number will be incremented by 1 when loaded by LKH
    '''
    V = A.number_of_nodes()
    N = V - A.graph['M']
    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    return '\n'.join((
        # first line is <number_of_nodes> <number_of_edges>
        f'{V} {2*A_nodes.number_of_edges() + 2*N}',
        # edges not including the depot (symmetric)
        '\n'.join('{s} {t} {cost:.0f}\n{t} {s} {cost:.0f}'.format(
            s=u+1, t=v+1, cost=scale*d)
                  for u, v, d in A_nodes.edges(data='length')),
        # depot connects to all nodes, but asymmetrically (zero-cost return)
        '\n'.join(f'{1} {n} {scale*d:.0f}\n{n} 1 0'
                  for n, d in enumerate(A.graph['d2roots'][:, 0], start=1)),
        ))


def lkh_acvrp(G_base: nx.Graph, *, capacity: int, time_limit: int,
              A: nx.Graph | None, scale: float = 1e4,
              vehicles: int | None = None, runs: int = 50,
              per_run_limit: float = 15.) -> nx.Graph:
    '''
    Lin-Kernighan-Helsgaun via LKH-3 binary.
    Asymmetric Capacitated Vehicle Routing Problem.

    Arguments:
        `G_base`: graph with the site's coordinates and boundary
        `capacity`: maximum vehicle capacity
        `time_limit`: [s] solver run time limit
        `A`: graph with allowed edges (if None, use complete graph)
        `scale`: factor to scale lengths
        `vehicles`: number of vehicles (if None, use the minimum feasible)
        `runs`: consult LKH manual
        `per_run_limit`: consult LKH manual
    '''
    G = nx.create_empty_copy(G_base)
    M = G.graph['M']
    assert M == 1, 'LKH allows only 1 depot'
    problem_fname = 'problem.txt'
    params_fname = 'params.txt'
    edge_fname = 'edge_file.txt'
    VertexC = G.graph['VertexC']
    N = VertexC.shape[0] - M
    w_saturation = np.iinfo(np.int32).max/1000
    weights, w_max = length_matrix_single_depot_from_G(A or G, scale)
    assert w_max <= w_saturation, 'ERROR: weight values outside int32 range.'
    weights = weights.clip(max=w_saturation).round().astype(np.int32)
    d2roots = G.graph.get('d2roots')
    if d2roots is None:
        d2roots = weights[0, 1:, None].copy()/scale
        G.graph['d2roots'] = d2roots
    if w_max > w_saturation:
        print('WARNING: at least one edge weight has been clipped.')
    #  weights = np.ones((N + M, N + M), dtype=int)
    with io.StringIO() as str_io:
        np.savetxt(str_io, weights, fmt='%d')
        edge_weights = str_io.getvalue()[:-1]

    #  print(edge_weights)
    output_fname = 'solution.out'
    specs = dict(
        NAME=G.graph.get('name', 'unnamed'),
        TYPE='ACVRP',  # maybe try asymmetric TSP: 'ATSP',
        # TYPE='ATSP',  # maybe try asymmetric capacitaded VRP: 'ACVRP',
        DIMENSION=N + M,  # CVRP number of nodes and depots
        # DIMENSION=N,  # TSP: number of nodes
        CAPACITY=capacity,
        EDGE_WEIGHT_TYPE='EXPLICIT',
        EDGE_WEIGHT_FORMAT='FULL_MATRIX',
    )
    data = dict(
        # DEMAND_SECTION='\n'.join(f'{i} 1' for i in range(N)) + f'\n{N} 0',
        DEPOT_SECTION=f'1\n-1',
        EDGE_WEIGHT_SECTION=edge_weights,
    )

    if False and A is not None:
        # TODO: Deprecate this. Passing edges always makes LKH fail.
        specs['EDGE_DATA_FORMAT'] = 'ADJ_LIST'
        A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
        data['EDGE_DATA_SECTION'] = '\n'.join((
            # depot has out-edges to all nodes
            f'1 {" ".join(str(n) for n in range(2, N + 2))} -1',
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
                '\n'.join(f'{n} 1\n1 {n}' for n in range(2, N + 2)),
                '-1\n'))
    if False and A is not None:
        # TODO: Deprecate this. EDGE_FILE is for the transformed problem.
        edge_str = make_edge_listing(A, scale)

    spec_str = '\n'.join(f'{k}: {v}' for k, v in specs.items())
    data_str = '\n'.join(f'{k}\n{v}' for k, v in data.items()) + '\nEOF'
    vehicles_min = math.ceil(N/capacity)
    if (vehicles is None) or (vehicles <= vehicles_min):
        # set to minimum feasible vehicle number
        if vehicles is not None and vehicles < vehicles_min:
            print(f'Vehicle number ({vehicles}) too low for feasibilty '
                  f'with capacity ({capacity}). Setting to {vehicles_min}.')
        vehicles = vehicles_min
        min_route_size = (N % capacity) or capacity
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
    if not penalty or result.stderr:
        print('===stdout===', result.stdout.decode('utf8'), sep='\n')
        print('===stderr===', result.stderr.decode('utf8'), sep='\n')
    else:
        #  tail = result.stdout[result.stdout.rfind(b'Successes/'):].decode('ascii')
        tail = result.stdout[result.stdout.rfind(b'Successes/'):].decode()
        entries = iter(tail.splitlines())
        # Decision to drop avg. stats: unreliable, possibly due to time_limit
        next(entries)  # skip sucesses line
        G.graph['cost_extrema'] = tuple(float(v) for v in re.match(
            r'Cost\.min = (-?\d+), Cost\.avg = -?\d+\.?\d*,'
            r' Cost\.max = -?(\d+)',
            next(entries)).groups())
        next(entries)  # skip gap line
        G.graph['penalty_extrema'] = tuple(float(v) for v in re.match(
            r'Penalty\.min = (\d+), Penalty\.avg = \d+\.?\d*,'
            r' Penalty\.max = (\d+)',
            next(entries)).groups())
        G.graph['trials_extrema'] = tuple(float(v) for v in re.match(
            r'Trials\.min = (\d+), Trials\.avg = \d+\.?\d*,'
            r' Trials\.max = (\d+)',
            next(entries)).groups())
        G.graph['runtime_extrema'] = tuple(float(v) for v in re.match(
            r'Time\.min = (\d+\.?\d*) sec., Time\.avg = \d+\.?\d* sec.,'
            r' Time\.max = (\d+\.?\d*) sec.',
            next(entries)).groups())
    d2roots = G.graph['d2roots']
    VertexC = G.graph['VertexC']
    nonAedges = []
    for branch in branches:
        if not branch:
            continue
        s = -1
        t = branch[0]
        G.add_edge(s, t, length=d2roots[t, s])
        #  print(f'{F[s]}–{F[t]}', end='')
        s = t
        for t in branch[1:]:
            #  print(f'–{F[t]}', end='')
            if A is not None and (s, t) in A.edges:
                G.add_edge(s, t, length=A[s][t]['length'])
            else:
                G.add_edge(s, t, length=np.hypot(*(VertexC[s] - VertexC[t]).T))
                nonAedges.append((s, t))
            s = t
    if branches:
        calcload(G)
    if nonAedges:
        G.graph['nonAedges'] = nonAedges
    else:
        PathFinder(G).create_detours(in_place=True)
    log = result.stdout.decode('utf8')
    G.graph.update(
        penalty=int(penalty),
        capacity=capacity,
        undetoured_length=float(minimum)/scale,
        edges_created_by='LKH-3',
        edges_fun=lkh_acvrp,
        creation_options=dict(
            complete=A is None,
            scale=scale,
            type=specs['TYPE'],
            time_limit=time_limit,
            runs=runs,
            per_run_limit=per_run_limit),
        runtime_unit='s',
        runtime=elapsed_time,
        solver_log=log,
        solution_time=_solution_time(log, minimum),
        fun_fingerprint=fun_fingerprint(),
    )
    return G


def _solution_time(log, undetoured_length) -> float:
    sol_repr = f'{undetoured_length}'
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


def get_sol_time(G: nx.Graph) -> float:
    """Graph must have graph attribute 'solver_log'"""
    log = G.graph['solver_log']
    sol = G.graph['undetoured_length']*G.graph['creation_options']['scale']
    sol_repr = f'{sol:.0f}'
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
