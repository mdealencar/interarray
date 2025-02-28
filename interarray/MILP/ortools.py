# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
from collections import defaultdict
from itertools import chain
import networkx as nx

from ortools.sat.python import cp_model

from .. import info
from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..interarraylib import fun_fingerprint, G_from_S
from ..pathfinding import PathFinder


class _SolutionStore(cp_model.CpSolverSolutionCallback):
    '''Ad hoc implementation of a callback that stores solutions to a pool.'''
    def __init__(self, model):
        super().__init__()
        self.solutions = []
        self.int_lits = (
            list(model.De.values())
            + list(model.Dg.values())
        )
        self.bool_lits = (
            list(model.Be.values())
            + list(model.Bg.values())
        )
        if hasattr(model, 'upstream'):
            self.bool_lits.extend(chain(
                *(v.values() for v in model.upstream.values())
            ))
    def on_solution_callback(self):
        solution = {var: self.boolean_value(var) for var in self.bool_lits}
        solution |= {var: self.value(var) for var in self.int_lits}
        self.solutions.append((self.objective_value, solution))


class CpSat(cp_model.CpSolver):
    '''
    This class wraps and changes the behavior of CpSolver in order to save all
    solutions found in a pool.

    THIS IS A HACK, it is meant to be used with `investigate_pool()` and
    nothing else.
    '''
    def solve(self, model: cp_model.CpModel) -> cp_model.cp_model_pb2.CpSolverStatus:
        '''
        Wrapper for CpSolver.solve() that saves the solutions.

        This method uses a custom CpSolverSolutionCallback to fill a solution
        pool stored in the attribute self.solutions.
        '''
        storer = _SolutionStore(model)
        result = super().solve(model, storer)
        self.solutions = storer.solutions
        self.num_solutions = len(storer.solutions)
        return result

    def load_solution(self, i: int) -> None:
        '''Select solution at position `i` in the pool.

        Indices start from 0 (last, aka best) and are ordered by increasing
        objective function value.
        It *only* affects methods `value`, `boolean_value` and `objective_value`.
        '''
        self._solution.objective_value, self._value_map = self.solutions[-i - 1]

    def boolean_value(self, literal: cp_model.IntVar) -> bool:
        return self._value_map[literal]

    def value(self, literal: cp_model.IntVar) -> int:
        return self._value_map[literal]


def investigate_pool(P: nx.PlanarEmbedding, A: nx.Graph,
                     model: cp_model.CpModel, solver: CpSat, result: int = 0) \
        -> nx.Graph:
    '''Go through the CpSat's solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    Λ = float('inf')
    num_solutions = solver.num_solutions
    info(f'Solution pool has {num_solutions} solutions.')
    for i in range(num_solutions):
        solver.load_solution(i)
        λ = solver.objective_value
        #  print(f'λ[{i}] = {λ}')
        if λ > Λ:
            info(f'Pool investigation over - next best undetoured length: {λ:.3f}')
            break
        S = S_from_solution(model, solver=solver)
        G = G_from_S(S, A)
        Hʹ = PathFinder(G, planar=P, A=A).create_detours()
        Λʹ = Hʹ.size(weight='length')
        if Λʹ < Λ:
            H, Λ = Hʹ, Λʹ
            info(f'Incumbent has (detoured) length: {Λ:.3f}')
    return H


def make_min_length_model(A: nx.Graph, capacity: int, *,
                          gateXings_constraint: bool = False,
                          gates_limit: bool | int = False,
                          branching: bool = True) -> cp_model.CpModel:
    '''
    Build ILP CP OR-tools model for the collector system length minimization.
    `A` is the graph with the available edges to choose from.

    `capacity`: cable capacity

    `gateXing_constraint`: if gates and edges are forbidden to cross.

    `gates_limit`: if True, use the minimum feasible number of gates
    (total for all roots); if False, no limit is imposed; if a number,
    use it as the limit.

    `branching`: if root branches are paths (False) or can be trees (True).
    '''
    R = A.graph['R']
    T = A.graph['T']
    d2roots = A.graph['d2roots']

    # Prepare data from A
    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    W = sum(w for _, w in A_nodes.nodes(data='power', default=1))
    E = tuple(((u, v) if u < v else (v, u))
              for u, v in A_nodes.edges())
    G = tuple((r, n) for n in A_nodes.nodes for r in range(-R, 0))
    w_E = tuple(A[u][v]['length'] for u, v in E)
    w_G = tuple(d2roots[n, r] for r, n in G)

    # Begin model definition
    m = cp_model.CpModel()

    # Parameters
    # k = m.NewConstant(3)
    k = capacity

    #############
    # Variables #
    #############

    # Binary edge present
    Be = {e: m.NewBoolVar(f'E_{e}') for e in E}
    # Binary gate present
    Bg = {e: m.NewBoolVar(f'G_{e}') for e in G}
    # Integer demand on edges
    De = {e: m.NewIntVar(-k + 1, k - 1, f'D_{e}') for e in E}
    # Integer demand on gates
    Dg = {e: m.NewIntVar(0, k, f'Dg_{e}') for e in G}

    ###############
    # Constraints #
    ###############

    # limit on number of gates
    min_gates = math.ceil(T/k)
    min_gate_load = 1
    if gates_limit:
        if isinstance(gates_limit, bool) or gates_limit == min_gates:
            # fixed number of gates
            m.Add((sum(Bg[r, u] for r in range(-R, 0)
                       for u in A_nodes.nodes.keys())
                   == math.ceil(T/k)))
            min_gate_load = T % k
        else:
            assert min_gates < gates_limit, (
                    f'Infeasible: T/k > gates_limit (T = {T}, k = {k},'
                    f' gates_limit = {gates_limit}).')
            # number of gates within range
            m.AddLinearConstraint(
                sum(Bg[r, u] for r in range(-R, 0)
                    for u in A_nodes.nodes.keys()),
                min_gates,
                gates_limit)
    else:
        # valid inequality: number of gates is at least the minimum
        m.Add(min_gates <= sum(Bg[r, n]
                               for r in range(-R, 0)
                               for n in A_nodes.nodes.keys()))

    # link edges' demand and binary
    for e in E:
        m.Add(De[e] == 0).OnlyEnforceIf(Be[e].Not())
        m.AddLinearExpressionInDomain(
            De[e],
            cp_model.Domain.FromIntervals([[-k + 1, -1], [1, k - 1]])
        ).OnlyEnforceIf(Be[e])

    # link gates' demand and binary
    for n in A_nodes.nodes.keys():
        for r in range(-R, 0):
            m.Add(Dg[r, n] == 0).OnlyEnforceIf(Bg[r, n].Not())
            m.Add(Dg[r, n] >= min_gate_load).OnlyEnforceIf(Bg[r, n])

    # total number of edges must be equal to number of non-root nodes
    m.Add(sum(Be.values()) + sum(Bg.values()) == T)

    # gate-edge crossings
    if gateXings_constraint:
        for e, g in gateXing_iter(A):
            m.AddAtMostOne(Be[e], Bg[g])

    # edge-edge crossings
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        m.AddAtMostOne(Be[u, v] for u, v in Xing)

    # flow consevation at each node
    for u in A_nodes.nodes.keys():
        m.Add(sum(De[u, v] if u < v else -De[v, u]
                  for v in A_nodes.neighbors(u))
              + sum(Dg[r, u] for r in range(-R, 0))
              == A.nodes[u].get('power', 1))

    if not branching:
        # non-branching (limit the nodes' degrees to 2)
        for u in A_nodes.nodes.keys():
            m.Add(sum((Be[u, v] if u < v else Be[v, u])
                      for v in A_nodes.neighbors(u))
                  + sum(Bg[r, u] for r in range(-R, 0)) <= 2)
            # each node is connected to a single root
            m.AddAtMostOne(Bg[r, u] for r in range(-R, 0))
    else:
        # If degree can be more than 2, enforce only one
        # edge flowing towards the root (up).

        # This is utterly complicated when compared to
        # the pyomo model that uses directed edges.
        # OR-tools could use directed edges too
        # (this all began as an experiment).
        upstream = defaultdict(dict)
        for u, v in E:
            direct = m.NewBoolVar(f'up_{u, v}')
            reverse = m.NewBoolVar(f'down_{u, v}')
            # channeling
            m.Add(De[u, v] > 0).OnlyEnforceIf(direct)
            m.Add(De[u, v] <= 0).OnlyEnforceIf(direct.Not())
            upstream[u][v] = direct
            m.Add(De[u, v] < 0).OnlyEnforceIf(reverse)
            m.Add(De[u, v] >= 0).OnlyEnforceIf(reverse.Not())
            upstream[v][u] = reverse
        for n in A_nodes.nodes.keys():
            # single root enforcement is encompassed here
            m.AddAtMostOne(
                *upstream[n].values(), *tuple(Bg[r, n] for r in range(-R, 0))
            )
        m.upstream = upstream

    # assert all nodes are connected to some root (using gate edge demands)
    m.Add(sum(Dg[r, n] for r in range(-R, 0)
              for n in A_nodes.nodes.keys()) == W)

    #############
    # Objective #
    #############

    m.Minimize(cp_model.LinearExpr.WeightedSum(Be.values(), w_E)
               + cp_model.LinearExpr.WeightedSum(Bg.values(), w_G))

    # save data structure as model attributes
    m.Be, m.Bg, m.De, m.Dg, m.R, m.T, m.k = Be, Bg, De, Dg, R, T, k
    #  m.site = {key: A.graph[key]
    #            for key in ('T', 'R', 'B', 'VertexC', 'border', 'obstacles',
    #                        'name', 'handle')
    #            if key in A.graph}

    ##################
    # Store metadata #
    ##################

    m.handle = A.graph['handle']
    m.name = A.graph.get('name', 'unnamed')
    m.method_options = dict(gateXings_constraint=gateXings_constraint,
                            gates_limit=gates_limit,
                            branching=branching)
    m.fun_fingerprint = fun_fingerprint()
    m.warmed_by = None
    return m


def warmup_model(model: cp_model.CpModel, S: nx.Graph) -> cp_model.CpModel:
    '''
    Changes `model` in-place.

    Only implemented for non-branching models.
    '''
    model.ClearHints()
    upstream = getattr(model, 'upstream', None)
    branched = upstream is not None
    for (u, v), Be in model.Be.items():
        is_in_G = (u, v) in S.edges
        model.AddHint(Be, is_in_G)
        De = model.De[u, v]
        if is_in_G:
            edgeD = S.edges[u, v]
            reverse = edgeD['reverse']
            model.AddHint(De, edgeD['load']*(1 if reverse else -1))
            if branched:
                model.AddHint(upstream[u][v], reverse)
                model.AddHint(upstream[v][u], not reverse)
        else:
            model.AddHint(De, 0)
            if branched:
                model.AddHint(upstream[u][v], False)
                model.AddHint(upstream[v][u], False)
    for rn, Bg in model.Bg.items():
        is_in_G = rn in S.edges
        model.AddHint(Bg, is_in_G)
        Dg = model.Dg[rn]
        model.AddHint(Dg, S.edges[rn]['load'] if is_in_G else 0)
    model.warmed_by = S.graph['creator']
    return model


def S_from_solution(model: cp_model.CpModel,
                    solver: cp_model.CpSolver, result: int = 0) -> nx.Graph:
    '''Create a topology `S` from the OR-tools solution to the MILP model.

    Args:
        model: passed to the solver.
        solver: used to solve the model.
        result: irrelevant, exists only to mirror the Pyomo alternative.
    Returns:
        Graph topology from the solution.
    '''
    # the solution is in the solver object not in the model

    # Metadata
    R, T = model.R, model.T
    solver_name = 'ortools'
    bound = solver.best_objective_bound
    objective = solver.objective_value
    S = nx.Graph(
        R=R, T=T,
        handle=model.handle,
        capacity=model.k,
        objective=objective,
        bound=bound,
        runtime=solver.wall_time,
        termination=solver.status_name(),
        gap=1. - bound/objective,
        creator='MILP.' + solver_name,
        has_loads=True,
        method_options=dict(
            solver_name=solver_name,
            mipgap=solver.parameters.relative_gap_limit,
            timelimit=solver.parameters.max_time_in_seconds,
            fun_fingerprint=model.fun_fingerprint,
            **model.method_options,
        ),
        solver_details=dict(
            strategy=solver.solution_info(),
        )
    )

    if model.warmed_by is not None:
        S.graph['warmstart'] = model.warmed_by

    # Graph data
    # gates
    gates_and_loads = tuple((r, n, solver.value(model.Dg[r, n]))
                            for (r, n), bg in model.Bg.items()
                            if solver.boolean_value(bg))
    S.add_weighted_edges_from(gates_and_loads, weight='load')
    # node-node edges
    S.add_weighted_edges_from(
        ((u, v, abs(solver.value(model.De[u, v])))
         for (u, v), be in model.Be.items()
         if solver.boolean_value(be)),
        weight='load'
    )

    # set the 'reverse' edges property
    # node-node edges
    nx.set_edge_attributes(
        S,
        {(u, v): solver.value(model.De[u, v]) > 0
         for (u, v), be in model.Be.items() if solver.boolean_value(be)},
        name='reverse')

    # propagate loads from edges to nodes
    subtree = -1
    for r in range(-R, 0):
        for u, v in nx.edge_dfs(S, r):
            S.nodes[v]['load'] = S.edges[u, v]['load']
            if u == r:
                subtree += 1
            S.nodes[v]['subtree'] = subtree
        rootload = 0
        for nbr in S.neighbors(r):
            # set the 'reverse' edge attribute for gates
            S[r][nbr]['reverse'] = False
            rootload += S.nodes[nbr]['load']
        S.nodes[r]['load'] = rootload
    return S
