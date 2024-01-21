'''
Module to help writing scripts for programatic job submission to the cluster
(LSF-based). This module needs some polishing.
'''

import json
import math
import os
import sys
import traceback
import subprocess

import dill
from pony.orm import db_session
import numpy as np

from interarray import new_dbmodel as dbmodel
from interarray.new_storage import packmethod, packnodes
from interarray.pathfinding import PathFinder
from interarray.MILP.pyomo import MILP_solution_to_G as cplex_MILP_solution_to_G
from interarray.interarraylib import fun_fingerprint
from interarray.geometric import make_graph_metrics


class DBhelper():
    '''
    This class has a single exposed method: .is_in_database()
    Its purpose is to avoid running cases already stored in the database.
    '''

    def __init__(self, database):
        # initialize database connection
        self.db = dbmodel.open_database(database)

    def is_in_database(self, G, capacity, edges_fun, method_options):
        # Make sure nodeset is compatible with the database.
        # There have been some issues with same nodeset
        # resulting in different digests and being added twice
        # (program fails because nodeset.name must be unique).
        nodes_pack = packnodes(G)
        method_pack = packmethod(fun_fingerprint(edges_fun), method_options)
        with db_session:
            nodes_entry = self.db.NodeSet.get(name=nodes_pack['name'])
        if nodes_entry is not None:
            print(f'NodeSet found in the database: {nodes_entry.name}')
            # if NodeSet name already in use, but digest is different do not
            # allow going forward
            if nodes_entry.digest != nodes_pack['digest']:
                print('Error: nodeset name <' + nodes_pack['name']
                      + '> already in use.', file=sys.stderr)
                return True

            # check if there is a result for the same pair (NodeSet, Method)
            with db_session:
                if self.db.Method.exists(digest=method_pack['digest']):
                    print('Method found in the database.')
                    if self.db.EdgeSet.exists(
                            nodes=self.db.NodeSet[nodes_entry.digest],
                            method=self.db.Method[method_pack['digest']],
                            capacity=capacity):
                        # TODO: decide whether to allow it to run if machine is
                        #       different. For the moment, not allowing.
                        print('Skipping: result is already in the database. '
                              'Exiting...', file=sys.stderr)
                        return True
        return False


def from_environment():
    method_options = json.loads(os.environ['INTERARRAY_METHOD'])
    problem_options = json.loads(os.environ['INTERARRAY_PROBLEM'])
    database = os.environ['INTERARRAY_DATABASE']
    return method_options, problem_options, database


def to_environment(problem_options: dict, method_options: dict,
                   database: str):
    os.environ['INTERARRAY_METHOD'] = json.dumps(method_options)
    os.environ['INTERARRAY_PROBLEM'] = json.dumps(problem_options)
    os.environ['INTERARRAY_DATABASE'] = database
    return ['INTERARRAY_METHOD', 'INTERARRAY_PROBLEM', 'INTERARRAY_DATABASE']


def dict_from_solver_status(solver_name, solver, status):
    keys = ('bound', 'objective', 'MILPtermination', 'runtime')
    if solver_name == 'gurobi':
        return dict(zip(keys, [
            solver.results['Problem'][0]['Lower bound'],
            solver.results['Problem'][0]['Upper bound'],
            solver.results['Solver'][0]['Termination condition'].value,
            solver.results['Solver'][0]['Wallclock time'],
            ]))
    elif solver_name == 'cplex' or solver_name == 'beta':
        return dict(zip(keys, [
            status['Problem'][0]['Lower bound'],
            status['Problem'][0]['Upper bound'],
            status['Solver'][0]['Termination condition'],
            status['Solver'][0]['Wallclock time'],
            ]))
    elif solver_name == 'ortools':
        return dict(zip(keys, [
            solver.BestObjectiveBound(),
            solver.ObjectiveValue(),
            solver.StatusName(),
            solver.WallTime(),
            ]))


def cplex_load_solution_from_pool(solver, soln):
    cplex = solver._solver_model
    vals = cplex.solution.pool.get_values(soln)
    vars_to_load = solver._pyomo_var_to_ndx_map.keys()
    for pyomo_var, val in zip(vars_to_load, vals):
        if solver._referenced_variables[pyomo_var] > 0:
            pyomo_var.set_value(val, skip_validation=True)


def cplex_investigate_pool(A, G, m, solver, info2store):
    '''Go through the CPLEX solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    # process the best layout first
    H = try_pathfinding_with_exc_handling(info2store, solver, G)
    H_incumbent = H
    L_incumbent = H.size(weight='length')
    print(f'First incumbent has length: {L_incumbent:.3f}')
    # now check the additional layouts
    cplex = solver._solver_model
    # G was generated with the first solution in the sorted Pool: skip it
    Pool = sorted((cplex.solution.pool.get_objective_value(i), i)
                  for i in range(cplex.solution.pool.get_num()))[1:]
    print(f'Solution pool has {len(Pool)} solutions')
    for L_pool, soln in Pool:
        if L_incumbent < L_pool:
            print('Finished analyzing solution pool.')
            break
        cplex_load_solution_from_pool(solver, soln)
        G = cplex_MILP_solution_to_G(m, solver, A)
        H = try_pathfinding_with_exc_handling(info2store, solver, G)
        L_contender = H.size(weight='length')
        if L_contender < L_incumbent:
            L_incumbent = L_contender
            H_incumbent = H
            print(f'New incumbent found with length: {L_incumbent:.3f}')
    return H_incumbent


def cplex_investigate_pool_inplace(A, G, m, solver, info2store):
    '''Go through the CPLEX solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    # process the best layout first
    try_pathfinding_with_exc_handling(info2store, solver, G, in_place=True)
    G_incumbent = G
    L_incumbent = G.size(weight='length')
    print(f'First incumbent has length: {L_incumbent:.0f}')
    # now check the additional layouts
    cplex = solver._solver_model
    # G was generated with the first solution in the sorted Pool: skip it
    Pool = sorted((cplex.solution.pool.get_objective_value(i), i)
                  for i in range(cplex.solution.pool.get_num()))[1:]
    print(f'Solution pool has {len(Pool)} solutions')
    for L_pool, soln in Pool:
        if L_incumbent < L_pool:
            print('Finished analyzing solution pool.')
            break
        cplex_load_solution_from_pool(solver, soln)
        G = cplex_MILP_solution_to_G(m, solver, A)
        try_pathfinding_with_exc_handling(info2store, solver, G, in_place=True)
        L_contender = G.size(weight='length')
        if L_contender < L_incumbent:
            L_incumbent = L_contender
            G_incumbent = G
            print(f'New incumbent found with length: {L_incumbent:.3f}')
    return G_incumbent


def try_pathfinding_with_exc_handling(info2store, solver, G, in_place=False):
    dumpgraphs = False
    print('Instantiating PathFinder...')
    try:
        pf = PathFinder(G)
        print('Creating detours...')
        try:
            H = pf.create_detours(in_place=in_place)
        except Exception:
            traceback.print_exc()
            # print(f'Exception "{exc}" caught while creating detours.',
            #       file=sys.stderr)
            dumpgraphs = True
            partial_solution = True
    except Exception:
        traceback.print_exc()
        # print(f'Exception "{exc}" caught while instantiating PathFinder.',
        #       file=sys.stderr)
        dumpgraphs = True
        partial_solution = False

    if dumpgraphs:
        job_id = info2store['solver_details']['job_id']
        solver_name = info2store['creation_options']['solver_name']
        handle = info2store['handle']
        capacity = info2store['capacity']
        dumpname = f'dump_{job_id}_{solver_name}_{handle}_{capacity}'
        G_fname = dumpname + '_G.dill'
        print(f'Dumping G graph to <{G_fname}>.', file=sys.stderr)
        dill.dump(G, open(G_fname, 'wb'))
        if partial_solution:
            H_fname = dumpname + '_H.dill'
            print(f'Dumping H graph to <{H_fname}>.', file=sys.stderr)
            dill.dump(pf.H, open(H_fname, 'wb'))
        if solver_name == 'gurobi':
            solver.close()
        elif solver_name == 'cplex':
            solver._solver_model.end()
        elif solver_name == 'beta':
            solver._solver_model.end()
        exit(1)

    return H


def memory_usage_model_MB(N, solver_name):
    mem = 500*N + 0.8*N**2
    if solver_name == 'cplex':
        return round(mem)
    elif solver_name == 'beta':
        return round(mem)
    else:
        return round(mem/3)


def unify_roots(G_base):
    '''
    `G_base` is changed in-place.

    Modify the provided nx.Graph `G_base` by reducing its root node to one.
        - nonroot nodes and boundary of `G_base` are not changed;
        - root nodes of `G_base` are replaced by a single root that is the
          centroid of the original ones.
    '''
    M = G_base.graph['M']
    if M <= 1:
        return
    N = G_base.number_of_nodes() - M
    VertexC = G_base.graph['VertexC']
    G_base.remove_nodes_from(range(-M, -1))
    G_base.graph['VertexC'] = np.r_[
            VertexC[:N],
            VertexC[-M:].mean(axis=0)[np.newaxis, :]
            ]
    G_base.graph['M'] = M = 1
    G_base.graph['name'] += '.1_OSS'
    G_base.graph['handle'] += '_1'
    make_graph_metrics(G_base)
    return


solver_options = {}
# Solver's settings
# Gurobi
solver_options['gurobi'] = dict(
    factory=dict(
        _name='gurobi',
        solver_io='python',),
)
# CPLEX
solver_options['cplex'] = dict(
    factory=dict(
        _name='cplex',
        solver_io='python',),
    # threshold for switching node storage strategy
    workmem=30000,  # in MB
    # node storage file switch (activates when workmem is exceeded):
    #   0) in-memory
    #   1) (the default) fast compression algorithm in-memory
    #   2) write to disk
    #   3) write to disk and compress
    mip_strategy_file=3,
    # tree memory limit:
    #   limit the size of the tree so that it does not exceed available disk
    #   space, when you choose settings 2 or 3 in the node storage file switch
    mip_limits_treememory=50000,  # in MB
    # directory for working files (if ommited, uses $TMPDIR)
    #  workdir='/tmp',
    workdir=(os.environ.get('TMPDIR')
             or os.environ.get('TMP')
             or '/tmp'),
)
solver_options['beta'] = solver_options['cplex']
# Google OR-Tools CP-SAT
solver_options['ortools'] = {}


class CondaJob:
    '''
    `mem_per_core` and `max_mem` in MB
    `time_limit` must be datetime.timedelta
    '''

    def __init__(self, cmdlist, *, conda_env, queue_name, jobname,
                 mem_per_core, max_mem, cores, time_limit, email=None,
                 cwd=None, env_variables=None):
        self.jobscript = \
            f'''#!/usr/bin/env sh
            ## queue
            #BSUB -q {queue_name}
            ## job name
            #BSUB -J {jobname}
            ## cores
            #BSUB -n {cores}
            ## cores must be on the same host
            #BSUB -R "span[hosts=1]"
            ## RAM per core/slot
            #BSUB -R "rusage[mem={round(mem_per_core)}MB]"
            ## job termination threshold: RAM per core/slot (Resident set size)
            #BSUB -M {math.ceil(max_mem)}MB
            ## job termination threshold: execution time (hh:mm)
            #BSUB -W {':'.join(str(time_limit).split(':')[:2])}
            ## stdout
            #BSUB -o {jobname}_%J.out
            ## stderr
            #BSUB -e {jobname}_%J.err
            '''
        if cwd is not None:
            self.jobscript += \
                f'''## job's current working directory
                #BSUB -cwd {cwd}
                '''
        if env_variables is not None:
            self.jobscript += \
                f''' ## environment variables to propagate
                #BSUB -env {",".join(env_variables)}
                '''
        if email is not None:
            self.jobscript += \
                f'''## email
                #BSUB -u {email}
                ## notify on end
                #BSUB -N
                '''
        self.jobscript += ' '.join(
            [os.environ['CONDA_EXE'], 'run', '--no-capture-output',
             '-n', conda_env] + cmdlist)
        self.summary = \
            f'''submitted: {jobname} (# of cores: {cores}, memory: \
            {mem_per_core*cores/1000:.1f} GB, time limit: {time_limit})'''

    def run(self, quiet=False):
        subprocess.run(['bsub'], input=self.jobscript.encode())
        if not quiet:
            print(self.summary)
