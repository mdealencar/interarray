# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
from collections import namedtuple, defaultdict
import networkx as nx

import pyomo.environ as pyo
from pyomo.contrib.solver.base import SolverBase
from pyomo.opt import SolverResults

from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..interarraylib import fun_fingerprint, G_from_S
from ..pathfinding import PathFinder


# solver option name mapping (pyomo should have taken care of this)
_common_options = namedtuple('common_options', 'mipgap timelimit')
_optname = defaultdict(lambda: _common_options(*_common_options._fields))
_optname['cbc'] = _common_options('ratioGap', 'seconds')
_optname['highs'] = _common_options('mip_rel_gap', 'time_limit')
_optname['scip'] = _common_options('limits/gap', 'limits/time')


def make_min_length_model(A: nx.Graph, capacity: int, *,
                          gateXings_constraint: bool = False,
                          gates_limit: bool | int = False,
                          branching: bool = True) -> pyo.ConcreteModel:
    '''
    Build ILP Pyomo model for the collector system length minimization.
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
    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    W = sum(w for _, w in A_nodes.nodes(data='power', default=1))

    # Create model
    m = pyo.ConcreteModel()

    # Sets

    # the model uses directed edges (except for gate edges), so a duplicate
    # set of edges is created with the reversed tuples
    E = tuple(((u, v) if u < v else (v, u))
              for u, v in A_nodes.edges())
    Eʹ = tuple((v, u) for u, v in E)

    m.diE = pyo.Set(initialize=E + Eʹ)
    #  m.T = pyo.RangeSet(0, T - 1)
    m.T = pyo.Set(initialize=A_nodes.nodes.keys())
    m.R = pyo.RangeSet(-R, -1)

    ##############
    # Parameters #
    ##############

    m.d = pyo.Param(m.diE,
                    domain=pyo.PositiveReals,
                    name='edge_cost',
                    initialize=lambda m, u, v: A.edges[(u, v)]['length'])

    m.g = pyo.Param(m.R, m.T,
                    domain=pyo.PositiveReals,
                    name='gate_cost',
                    initialize=lambda m, r, n: d2roots[n, r])

    m.k = pyo.Param(domain=pyo.PositiveIntegers,
                    name='capacity', default=capacity)

    #############
    # Variables #
    #############

    m.Be = pyo.Var(m.diE, domain=pyo.Binary, initialize=0)
    m.De = pyo.Var(m.diE, domain=pyo.NonNegativeIntegers,
                   bounds=(0, m.k - 1), initialize=0)

    m.Bg = pyo.Var(m.R, m.T,
                   domain=pyo.Binary,
                   initialize=0)
    m.Dg = pyo.Var(m.R, m.T,
                   domain=pyo.NonNegativeIntegers,
                   bounds=(0, m.k),
                   initialize=0)

    ###############
    # Constraints #
    ###############

    # total number of edges must be equal to number of non-root nodes
    m.cons_edges_eq_nodes = pyo.Constraint(
        rule=lambda m: (sum(m.Be[u, v] for u, v in m.diE)
                        + sum(m.Bg[r, n] for r in m.R for n in m.T) == T)
    )

    # enforce a single directed edge between each node pair
    m.cons_one_diEdge = pyo.Constraint(
        E,
        rule=lambda m, u, v: m.Be[u, v] + m.Be[v, u] <= 1
    )

    # each node is connected to a single root
    m.cons_one_root = pyo.Constraint(
        m.T,
        rule=lambda m, n: sum(m.Bg[:, n]) <= 1)

    # gate-edge crossings
    if gateXings_constraint:
        m.cons_gateXedge = pyo.Constraint(
            gateXing_iter(A),
            rule=lambda m, u, v, r, n: (m.Be[u, v]
                                        + m.Be[v, u]
                                        + m.Bg[r, n] <= 1)
        )

    # edge-edge crossings
    def edgeXedge_rule(m, *vertices):
        lhs = sum((m.Be[u, v] + m.Be[v, u])
                  for u, v in zip(vertices[::2],
                                  vertices[1::2]))
        return lhs <= 1
    doubleXings = []
    tripleXings = []
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        if len(Xing) == 2:
            doubleXings.append(Xing)
        else:
            tripleXings.append(Xing)
    if doubleXings:
        m.cons_edgeXedge = pyo.Constraint(doubleXings,
                                          rule=edgeXedge_rule)
    if tripleXings:
        m.cons_edgeXedgeXedge = pyo.Constraint(tripleXings,
                                               rule=edgeXedge_rule)

    # bind binary active flags to demand
    m.cons_edge_active_iff_demand_lb = pyo.Constraint(
        m.diE,
        rule=lambda m, u, v: m.De[(u, v)] <= m.k*m.Be[(u, v)]
    )
    m.cons_edge_active_iff_demand_ub = pyo.Constraint(
        m.diE,
        rule=lambda m, u, v: m.Be[(u, v)] <= m.De[(u, v)]
    )
    m.cons_gate_active_iff_demand_lb = pyo.Constraint(
        m.R, m.T,
        rule=lambda m, r, n: m.Dg[r, n] <= m.k*m.Bg[r, n]
    )
    m.cons_gate_active_iff_demand_ub = pyo.Constraint(
        m.R, m.T,
        rule=lambda m, r, n: m.Bg[r, n] <= m.Dg[r, n]
    )

    # TODO: multiple cable types - THAT INVOLVES CHANGING m.Be
    # code below is a quick draft (does NOT work at all)
    #  if False:
    #      m.T = pyo.Set(range_number_of_cable_types)
    #      m.Bte = pyo.Var(m.E, m.T)
    #      # constraints
    #      for edge in E:
    #          # sums over t in T
    #          pyo.summation(m.Bte[edge]) + pyo.summation(m.Bte[edgeʹ]) <= 1
    #      for edge in E + Eʹ:
    #          # sums over t in T
    #          pyo.summation(m.Bte[edge]) <= m.De[edge]
    #          pyo.summation(m.k[t]*m.Bte[edge][t]) >= m.De[edge]

    # flow conservation with possibly non-unitary node power
    m.cons_flow_conservation = pyo.Constraint(
        m.T,
        rule=lambda m, u: (sum((m.De[u, v] - m.De[v, u])
                               for v in A_nodes.neighbors(u))
                           + sum(m.Dg[r, u] for r in m.R)
                           == A.nodes[u].get('power', 1))
    )

    # gates limit
    if gates_limit:
        def gates_limit_eq_rule(m):
            return (sum(m.Bg[r, u] for r in m.R for u in m.T)
                    == math.ceil(T/m.k))

        def gates_limit_ub_rule(m):
            return (sum(m.Bg[r, u] for r in m.R for u in m.T)
                    <= gates_limit)

        m.gates_limit = pyo.Constraint(rule=(gates_limit_eq_rule
                                             if isinstance(gates_limit, bool)
                                             else gates_limit_ub_rule))

    # non-branching
    if not branching:
        # just need to limit incoming edges since the outgoing are
        # limited by the m.cons_one_out_edge
        m.non_branching = pyo.Constraint(
            m.T,
            rule=lambda m, u: (sum(m.Be[v, u] for v in A_nodes.neighbors(u))
                               <= 1)
        )

    # assert all nodes are connected to some root
    m.cons_all_nodes_connected = pyo.Constraint(
        rule=lambda m: sum(m.Dg[r, n] for r in m.R for n in m.T) == W
    )

    # valid inequalities
    m.cons_min_gates_required = pyo.Constraint(
        rule=lambda m: (sum(m.Bg[r, n] for r in m.R for n in m.T)
                        >= math.ceil(T/m.k))
    )
    m.cons_incoming_demand_limit = pyo.Constraint(
        m.T,
        rule=lambda m, u: (sum(m.De[v, u] for v in A_nodes.neighbors(u))
                           <= m.k - 1)
    )
    m.cons_one_out_edge = pyo.Constraint(
        m.T,
        rule=lambda m, u: (sum(m.Be[u, v] for v in A_nodes.neighbors(u))
                           + sum(m.Bg[r, u] for r in m.R) == 1)
    )

    #############
    # Objective #
    #############

    m.length = pyo.Objective(
        expr=lambda m: pyo.sum_product(m.d, m.Be) + pyo.sum_product(m.g, m.Bg),
        sense=pyo.minimize,
    )

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


def warmup_model(model: pyo.ConcreteModel, S: nx.Graph) \
        -> pyo.ConcreteModel:
    '''
    Changes `model` in-place.
    '''
    Ne = len(model.diE)//2
    T = len(model.T)
    # the first half of diE has all the edges with u < v
    for u, v in list(model.diE)[:Ne]:
        if (u, v) in S.edges:
            if S[u][v]['reverse']:
                model.Be[u, v] = 1
                model.De[u, v] = S[u][v]['load']
            else:
                model.Be[v, u] = 1
                model.De[v, u] = S[u][v]['load']
    for r in model.R:
        for n in S.neighbors(r):
            model.Bg[r, n] = 1
            model.Dg[r, n] = S[n][r]['load']
    model.warmed_by = S.graph['creator']
    return model


def S_from_solution(model: pyo.ConcreteModel, solver: SolverBase,
                    result: SolverResults) -> nx.Graph:
    '''
    Create a topology `S` with the solution in `model` by `solver`.
    '''

    # Metadata
    R, T, k = len(model.R), len(model.T), model.k.value
    #  if 'highs.Highs' in str(solver):
    if hasattr(solver, 'highs_options'):
        solver_name = 'highs'
    elif solver.name.endswith('direct'):
        solver_name = solver.name[:-6].rstrip('_')
    elif solver.name.endswith('persistent'):
        solver_name = solver.name[:-10].rstrip('_')
    else:
        solver_name = solver.name
    bound = result['Problem'][0]['Lower bound']
    objective = result['Problem'][0]['Upper bound']
    # create a topology graph S from the solution
    S = nx.Graph(
        R=R, T=T,
        handle=model.handle,
        capacity=k,
        objective=objective,
        bound=bound,
        runtime=result['Solver'][0]['Wallclock time'],
        termination=result['Solver'][0]['Termination condition'].name,
        gap=1. - bound/objective,
        creator='MILP.pyomo.' + solver_name,
        has_loads=True,
        method_options=dict(
            solver_name=solver_name,
            mipgap=solver.options[_optname[solver_name].mipgap],
            timelimit=solver.options[_optname[solver_name].timelimit],
            fun_fingerprint=model.fun_fingerprint,
            **model.method_options,
        ),
        #  solver_details=dict(
        #  )
    )

    if model.warmed_by is not None:
        S.graph['warmstart'] = model.warmed_by

    # Graph data
    # gates
    S.add_weighted_edges_from(
        ((r, n, round(model.Dg[r, n].value))
         for (r, n), bg in model.Bg.items()
         if bg.value > 0.5),
        weight='load'
    )
    # node-node edges
    S.add_weighted_edges_from(
        ((u, v, round(model.De[u, v].value))
         for (u, v), be in model.Be.items()
         if be.value > 0.5),
        weight='load'
    )

    # set the 'reverse' edge attribute
    # node-node edges
    nx.set_edge_attributes(
        S,
        {(u, v): v > u for (u, v), be in model.Be.items() if be.value > 0.5},
        name='reverse')
    # propagate loads from edges to nodes
    subtree = -1
    for r in range(-R, 0):
        for u, v in nx.edge_dfs(S, r):
            S.nodes[v]['load'] = S[u][v]['load']
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


def gurobi_investigate_pool(P, A, model, solver, result):
    '''Go through the Gurobi's solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    # initialize incumbent total length
    solver_model = solver._solver_model
    Λ = float('inf')
    num_solutions = solver_model.getAttr('SolCount')
    print(f'Solution pool has {num_solutions} solutions.')
    # Pool = iter(sorted((cplex.solution.pool.get_objective_value(i), i)
                       # for i in range(cplex.solution.pool.get_num()))[1:])
    # model comes loaded with minimal-length undetoured solution
    for i in range(num_solutions):
        solver_model.setParam('SolutionNumber', i)
        λ = solver_model.getAttr('PoolObjVal')
        if λ > Λ:
            print(f'Pool investigation over - next best undetoured length: {λ:.3f}')
            break
        for omovar, gurvar in solver._pyomo_var_to_solver_var_map.items():
            omovar.set_value(round(gurvar.Xn), skip_validation=True)
        S = S_from_solution(model, solver=solver, result=result)
        G = G_from_S(S, A)
        Hʹ = PathFinder(G, planar=P, A=A).create_detours()
        Λʹ = Hʹ.size(weight='length')
        if Λʹ < Λ:
            H, Λ = Hʹ, Λʹ
            print(f'Incumbent has (detoured) length: {Λ:.3f}')
    return H


def cplex_load_solution_from_pool(solver, soln):
    cplex = solver._solver_model
    vals = cplex.solution.pool.get_values(soln)
    vars_to_load = solver._pyomo_var_to_ndx_map.keys()
    for pyomo_var, val in zip(vars_to_load, vals):
        if solver._referenced_variables[pyomo_var] > 0:
            pyomo_var.set_value(val, skip_validation=True)


def cplex_investigate_pool(P, A, model, solver, result):
    '''Go through the CPLEX solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    cplex = solver._solver_model
    # initialize incumbent total length
    Λ = float('inf')
    print(f'Solution pool has {cplex.solution.pool.get_num()} solutions.')
    Pool = iter(sorted((cplex.solution.pool.get_objective_value(i), i)
                       for i in range(cplex.solution.pool.get_num()))[1:])
    # model comes loaded with minimal-length undetoured solution
    while True:
        S = S_from_solution(model, solver=solver, result=result)
        G = G_from_S(S, A)
        Hʹ = PathFinder(G, planar=P, A=A).create_detours()
        Λʹ = Hʹ.size(weight='length')
        if Λʹ < Λ:
            H, Λ = Hʹ, Λʹ
            print(f'Incumbent has (detoured) length: {Λ:.3f}')
        # check if next best solution is worth processing
        try:
            λ, soln = next(Pool)
        except StopIteration:
            print('Pool exhausted.')
            break
        if λ > Λ:
            print(f'Done with pool - next best undetoured length: {λ:.3f}')
            break
        cplex_load_solution_from_pool(solver, soln)
    return H
