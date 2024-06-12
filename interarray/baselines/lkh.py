import math
import time
import os
import re
import io
import tempfile
import subprocess
from pathlib import Path
from typing import TextIO
import networkx as nx
import numpy as np

from . import weight_matrix_single_depot_from_G
from ..interarraylib import calcload
from ..pathfinding import PathFinder
from ..geometric import make_graph_metrics, delaunay


# TODO: Deprecate that. Unable to make LKH work in ACVRP with EDGE_FILE
#       It seems like EDGE_FILE is for candidate edges of the transformed
#       (symmetric) problem (i.e., LKH expects a higher node count)
def make_edge_listing(A: nx.Graph, precision_factor: float) -> str:
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
            s=u+1, t=v+1, cost=precision_factor*d)
                  for u, v, d in A_nodes.edges(data='length')),
        # depot connects to all nodes, but asymmetrically (zero-cost return)
        '\n'.join(f'{1} {n} {precision_factor*d:.0f}\n{n} 1 0'
                  for n, d in enumerate(A.graph['d2roots'][:, 0], start=1)),
        ))


def lkh_acvrp(G_base: nx.Graph, *, capacity: int, time_limit: int,
              runs: int = 50, precision_factor: float = 1e4, complete=False,
              per_run_limit: float = 15.) -> nx.Graph:
    G = nx.create_empty_copy(G_base)
    M = G.graph['M']
    assert M == 1, 'LKH allows only 1 depot'
    problem_fname = 'problem.txt'
    params_fname = 'params.txt'
    edge_fname = 'edge_file.txt'
    VertexC = G.graph['VertexC']
    N = VertexC.shape[0] - M
    weights, A = weight_matrix_single_depot_from_G(
            G, precision_factor=precision_factor, complete=complete)
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
    if False:
        specs['EDGE_DATA_FORMAT'] = 'ADJ_LIST'
        A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
        data['EDGE_DATA_SECTION'] = '\n'.join((
            # depot has out-edges to all nodes
            f'1 {" ".join(str(n) for n in range(2, N + 2))} -1',
            # all node have out-edges to depot
            '\n'.join(f'{n + 2} 1 {" ".join(str(a + 2) for a in adj)} -1'
                      for n, adj in A_nodes.adjacency()),
            '-1'
            ))
    #  if not complete:
    if False:  # adding edges just makes the solver fail
        specs['EDGE_DATA_FORMAT'] = 'EDGE_LIST'
        data['EDGE_DATA_SECTION'] = '\n'.join((
                '\n'.join(f'{u + 2} {v + 2}\n{v + 2} {u + 2}'
                          for u, v in A.edges if u >= 0 and v >= 0),
                '\n'.join(f'{n} 1\n1 {n}' for n in range(2, N + 2)),
                '-1\n'))

    spec_str = '\n'.join(f'{k}: {v}' for k, v in specs.items())
    data_str = '\n'.join(f'{k}\n{v}' for k, v in data.items()) + '\nEOF'
    params = dict(
        SPECIAL=None,  # None -> output only the key
        PRECISION=1000,  # d[i][j] = PRECISION*c[i][j] + pi[i] + pi[j]
        TOTAL_TIME_LIMIT=time_limit,
        TIME_LIMIT=per_run_limit,
        RUNS=runs,  # default: 10
        # MAX_TRIALS=100,  # default: number of nodes (DIMENSION)
        # TRACE_LEVEL=1,  # default is 1, 0 supresses output
        #  INITIAL_TOUR_ALGORITHM='GREEDY',  # { … | CVRP | MTSP | SOP } Default: WALK
        VEHICLES=math.ceil(N/capacity),
        MTSP_MIN_SIZE=(N % capacity) or capacity,
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
    #  edge_str = make_edge_listing(delaunay(G), precision_factor)
    with tempfile.TemporaryDirectory() as tmpdir:
        problem_fpath = os.path.join(tmpdir, problem_fname)
        #  edge_fpath = os.path.join(tmpdir, edge_fname)
        params_fpath = os.path.join(tmpdir, params_fname)
        params['PROBLEM_FILE'] = problem_fpath
        #  params['EDGE_FILE'] = edge_fpath
        params['MTSP_SOLUTION_FILE'] = os.path.join(tmpdir,
                                                    output_fname)
        Path(problem_fpath).write_text('\n'.join((spec_str, data_str)))
        Path(params_fpath).write_text('\n'.join(
            (f'{k} = {v}' if v is not None else k) for k, v in params.items()))
        #  Path(edge_fpath).write_text(edge_str)
        start_time = time.perf_counter()
        result = subprocess.run(['LKH', params_fpath], capture_output=True)
        elapsed_time = time.perf_counter() - start_time
        with open(os.path.join(tmpdir, output_fname), 'r') as f_sol:
            penalty, minimum = next(f_sol).split(':')[-1][:-1].split('_')
            next(f_sol)  # discard second line
            branches = [[int(node) - 2 for node in line.split(' ')[1:-5]]
                        for line in f_sol]
    #  print('===stdout===', result.stdout.decode('utf8'), sep='\n')
    #  print('===stderr===', result.stderr.decode('utf8'), sep='\n')
    tail = result.stdout[result.stdout.rfind(b'Successes/'):].decode('ascii')
    entries = iter(tail.splitlines())
    # Decision to drop avg. stats: unreliable, possibly due to time_limit
    next(entries)  # skip sucesses line
    G.graph['cost_extrema'] = tuple(float(v) for v in re.match(
        r'Cost\.min = (\d+), Cost\.avg = \d+\.?\d*, Cost\.max = (\d+)',
        next(entries)).groups())
    next(entries)  # skip gap line
    G.graph['penalty_extrema'] = tuple(float(v) for v in re.match(
        r'Penalty\.min = (\d+), Penalty\.avg = \d+\.?\d*,'
        r' Penalty\.max = (\d+)',
        next(entries)).groups())
    G.graph['trials_extrema'] = tuple(float(v) for v in re.match(
        r'Trials\.min = (\d+), Trials\.avg = \d+\.?\d*, Trials\.max = (\d+)',
        next(entries)).groups())
    G.graph['runtime_extrema'] = tuple(float(v) for v in re.match(
        r'Time\.min = (\d+\.?\d*) sec., Time\.avg = \d+\.?\d* sec.,'
        r' Time\.max = (\d+\.?\d*) sec.',
        next(entries)).groups())
    d2roots = G.graph['d2roots']
    VertexC = G.graph['VertexC']
    nonAedges = []
    if A is None:
        A = delaunay(G)
    for branch in branches:
        s = -1
        t = branch[0]
        G.add_edge(s, t, length=G.graph['d2roots'][t, s])
        #  print(f'{F[s]}–{F[t]}', end='')
        s = t
        for t in branch[1:]:
            #  print(f'–{F[t]}', end='')
            if (s, t) in A.edges:
                G.add_edge(s, t, length=A[s][t]['length'])
            else:
                G.add_edge(s, t, length=np.hypot(*(VertexC[s] - VertexC[t]).T))
                nonAedges.append((s, t))
            s = t
    calcload(G)
    if not nonAedges:
        PathFinder(G).create_detours(in_place=True)
    G.graph['nonAedges'] = nonAedges
    G.graph['penalty'] = int(penalty)
    G.graph['undetoured_length'] = minimum/precision_factor
    G.graph['capacity'] = capacity
    G.graph['edges_created_by'] = 'LKH-3'
    G.graph['edges_fun'] = lkh_acvrp
    G.graph['creation_options'] = dict(
            type=specs['TYPE'],
            time_limit=time_limit,
            runs=runs,
            per_run_limit=per_run_limit,
            precision_factor=precision_factor,
            complete=complete)
    G.graph['runtime_unit'] = 's'
    G.graph['runtime'] = elapsed_time
    return G
