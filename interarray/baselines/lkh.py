import math
import time
import os
import io
import tempfile
import subprocess
import networkx as nx
import numpy as np

from . import weight_matrix_single_depot_from_G
from ..interarraylib import calcload
from ..pathfinding import PathFinder
from ..geometric import make_graph_metrics, delaunay


def lkh_acvrp(G_base: nx.Graph, *, capacity: int, time_limit: int, runs=50,
              precision_factor: float = 1e8, complete=False) -> nx.Graph:
    G = nx.create_empty_copy(G_base)
    M = G.graph['M']
    assert M == 1  # LKH allows only 1 depot
    VertexC = G.graph['VertexC']
    N = VertexC.shape[0] - M
    weights, A = weight_matrix_single_depot_from_G(
            G, precision_factor=precision_factor, complete=complete)
    with io.StringIO() as str_io:
        np.savetxt(str_io, weights, fmt='%d')
        edge_weights = str_io.getvalue()[:-1]

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
    if not complete:
        specs['EDGE_DATA_FORMAT'] = 'EDGE_LIST'
        data['EDGE_DATA_SECTION'] = '\n'.join(f'{u + 2} {v + 2}'
                                              for u, v in A.edges) + '\n-1\n',

    spec_str = '\n'.join(f'{k}: {v}' for k, v in specs.items())
    data_str = '\n'.join(f'{k}\n{v}' for k, v in data.items()) + '\nEOF'
    params = dict(
        SPECIAL=None,  # None -> output only the key
        TIME_LIMIT=time_limit,
        RUNS=runs,  # default: 10
        # MAX_TRIALS=100,  # default: number of nodes (DIMENSION)
        # TRACE_LEVEL=1,  # default is 1, 0 supresses output
        # INITIAL_TOUR_ALGORITHM='CVRP',  # { … | CVRP | MTSP | SOP } Default: WALK
        VEHICLES=math.ceil(N/capacity),
        MTSP_MIN_SIZE=(N % capacity) or capacity,
        MTSP_MAX_SIZE=capacity,
        MTSP_OBJECTIVE='MINSUM',  # [ MINMAX | MINMAX_SIZE | MINSUM ]
        MTSP_SOLUTION_FILE=output_fname,
    )

    problem_fname = 'problem.txt'
    params_fname = 'params.txt'
    with tempfile.TemporaryDirectory() as tmpdir:
        problem_fpath = os.path.join(tmpdir, problem_fname)
        params_fpath = os.path.join(tmpdir, params_fname)
        params['PROBLEM_FILE'] = problem_fpath
        params['MTSP_SOLUTION_FILE'] = os.path.join(tmpdir,
                                                    output_fname)
        with open(problem_fpath, 'w') as f_problem:
            f_problem.write('\n'.join((spec_str, data_str)))
        with open(params_fpath, 'w') as f_params:
            f_params.write('\n'.join((f'{k} = {v}' if v is not None else k)
                                     for k, v in params.items()))
        start_time = time.perf_counter()
        result = subprocess.run(['LKH', params_fpath], capture_output=True)
        run_time = time.perf_counter() - start_time
        print('===stdout===', result.stdout.decode('utf8'), sep='\n')
        print('===stderr===', result.stderr.decode('utf8'), sep='\n')
        with open(os.path.join(tmpdir, output_fname), 'r') as f_sol:
            next(f_sol), next(f_sol)  # discard first two lines (unstructured)
            branches = [[int(node) - 2 for node in line.split(' ')[1:-5]]
                        for line in f_sol]
    d2roots = G.graph['d2roots']
    #  return G, branches
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
    PathFinder(G).create_detours(in_place=True)
    make_graph_metrics(G)
    G.graph['nonAedges'] = nonAedges
    # G.graph['iterations'] = ???
    G.graph['capacity'] = capacity
    # G.graph['overfed'] = [len(G[root])/np.ceil(N/capacity)*M
    #                       for root in roots]
    G.graph['edges_created_by'] = 'LKH-3'
    G.graph['edges_fun'] = lkh_acvrp
    G.graph['creation_options'] = dict(
            type=specs['TYPE'],
            time_limit=params['TIME_LIMIT'],
            runs=params['RUNS'],
            precision_factor=precision_factor,
            complete=complete)
    G.graph['runtime_unit'] = 's'
    G.graph['runtime'] = run_time
    return G
