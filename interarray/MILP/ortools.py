# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
from collections import defaultdict

import networkx as nx
import numpy as np

from ortools.sat.python import cp_model

from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..interarraylib import calcload, fun_fingerprint


def make_MILP_length(A, k, gateXings_constraint=False, gates_limit=False,
                     branching=True):
    '''
    MILP OR-tools CP model for the collector system optimization.
    A is the networkx graph with the available edges.

    `k`: cable capacity

    `gateXings_constraint`: whether to avoid crossing of gate edges.

    `gates_limit`: if True, use the minimum feasible number of gates.
    (total for all roots); if False, no limit is imposed; if a number,
    use it as the limit.

    `branching`: if True, allow subtrees to branch; if False, no branching.
    '''
    M = A.graph['M']
    N = A.graph['N']
    d2roots = A.graph['d2roots']

    # Prepare data from A
    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    W = sum(w for n, w in A_nodes.nodes(data='power', default=1))
    E = tuple(((u, v) if u < v else (v, u))
              for u, v in A_nodes.edges())
    G = tuple((r, n) for n in A_nodes.nodes.keys() for r in range(-M, 0))
    w_E = tuple(A[u][v]['length'] for u, v in E)
    w_G = tuple(d2roots[n, r] for r, n in G)

    # Begin model definition
    m = cp_model.CpModel()

    # Parameters
    # k = m.NewConstant(3)

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
    min_gates = math.ceil(N/k)
    min_gate_load = 1
    if gates_limit:
        if isinstance(gates_limit, bool) or gates_limit == min_gates:
            # fixed number of gates
            m.Add((sum(Bg[r, u] for r in range(-M, 0)
                       for u in A_nodes.nodes.keys())
                   == math.ceil(N/k)))
            min_gate_load = N % k
        else:
            assert min_gates < gates_limit, (
                    f'Infeasible: N/k > gates_limit (N = {N}, k = {k},'
                    f' gates_limit = {gates_limit}).')
            # number of gates within range
            m.AddLinearConstraint(
                sum(Bg[r, u] for r in range(-M, 0)
                    for u in A_nodes.nodes.keys()),
                min_gates,
                gates_limit)
    else:
        # valid inequality: number of gates is at least the minimum
        m.Add(min_gates <= sum(Bg[r, n]
                               for r in range(-M, 0)
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
        for r in range(-M, 0):
            m.Add(Dg[r, n] == 0).OnlyEnforceIf(Bg[r, n].Not())
            m.Add(Dg[r, n] >= min_gate_load).OnlyEnforceIf(Bg[r, n])

    # total number of edges must be equal to number of non-root nodes
    m.Add(sum(Be.values()) + sum(Bg.values()) == N)

    # gate-edge crossings
    if gateXings_constraint:
        for e, g in gateXing_iter(A):
            m.AddAtMostOne(Be[e], Bg[g])

    # edge-edge crossings
    for Xing in edgeset_edgeXing_iter(A):
        m.AddAtMostOne(Be[u, v] if u >= 0 else Bg[u, v]
                       for u, v in Xing)

    # flow consevation at each node
    for u in A_nodes.nodes.keys():
        m.Add(sum(De[u, v] if u < v else -De[v, u]
                  for v in A_nodes.neighbors(u))
              + sum(Dg[r, u] for r in range(-M, 0))
              == A.nodes[u].get('power', 1))

    if not branching:
        # non-branching (limit the nodes' degrees to 2)
        for u in A_nodes.nodes.keys():
            m.Add(sum((Be[u, v] if u < v else Be[v, u])
                      for v in A_nodes.neighbors(u))
                  + sum(Bg[r, u] for r in range(-M, 0)) <= 2)
            # each node is connected to a single root
            m.AddAtMostOne(Bg[r, u] for r in range(-M, 0))
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
                *upstream[n].values(), *tuple(Bg[r, n] for r in range(-M, 0))
            )
        m.upstream = upstream

    # assert all nodes are connected to some root (using gate edge demands)
    m.Add(sum(Dg[r, n] for r in range(-M, 0)
              for n in A_nodes.nodes.keys()) == W)

    #############
    # Objective #
    #############

    m.Minimize(cp_model.LinearExpr.WeightedSum(Be.values(), w_E)
               + cp_model.LinearExpr.WeightedSum(Bg.values(), w_G))

    # save data structure as model attributes
    m.Be, m.Bg, m.De, m.Dg, m.M, m.N, m.k = Be, Bg, De, Dg, M, N, k
    #  m.site = {key: A.graph[key]
    #            for key in ('N', 'M', 'B', 'VertexC', 'border', 'exclusions',
    #                        'name', 'handle')
    #            if key in A.graph}
    m.creation_options = dict(gateXings_constraint=gateXings_constraint,
                              gates_limit=gates_limit,
                              branching=branching)
    m.fun_fingerprint = fun_fingerprint()
    return m


def MILP_warmstart_from_T(m: cp_model.CpModel, T: nx.Graph):
    '''
    Only implemented for non-branching models.
    '''
    m.ClearHints()
    upstream = getattr(m, 'upstream', None)
    if upstream is not None:
        let_branch = True
    for (u, v), Be in m.Be.items():
        is_in_G = (u, v) in T.edges
        m.AddHint(Be, is_in_G)
        De = m.De[u, v]
        if is_in_G:
            edgeD = T.edges[u, v]
            m.AddHint(De, edgeD['load']*(1 if edgeD['reverse'] else -1))
        else:
            m.AddHint(De, 0)
    for rn, Bg in m.Bg.items():
        is_in_G = rn in T.edges
        m.AddHint(Bg, is_in_G)
        Dg = m.Dg[rn]
        m.AddHint(Dg, T.edges[rn]['load'] if is_in_G else 0)


def MILP_solution_to_T(model, *, solver):
    '''Translate a MILP OR-tools solution to a networkx graph.'''
    # the solution is in the solver object not in the model

    # create a topology graph T from the solution
    T = nx.Graph(
        M=model.M, N=model.N,
        capacity=model.k,
        edges_created_by='MILP.ortools',
        creation_options=model.creation_options,
        has_loads=True,
        fun_fingerprint=model.fun_fingerprint,
    )

    # gates
    gates_and_loads = tuple((r, n, solver.Value(model.Dg[r, n]))
                            for (r, n), bg in model.Bg.items()
                            if solver.BooleanValue(bg))
    T.add_weighted_edges_from(gates_and_loads, weight='load')
    # node-node edges
    T.add_weighted_edges_from(
        ((u, v, abs(solver.Value(model.De[u, v])))
         for (u, v), be in model.Be.items()
         if solver.BooleanValue(be)),
        weight='load'
    )

    # set the 'reverse' edges property
    # node-node edges
    nx.set_edge_attributes(
        T,
        {(u, v): solver.Value(model.De[u, v]) > 0
         for (u, v), be in model.Be.items() if solver.BooleanValue(be)},
        name='reverse')
    # gate edges
    for r in range(-M, 0):
        for n in T[r]:
            T[r][n]['reverse'] = False

    # propagate loads from edges to nodes
    subtree = -1
    for r in range(-M, 0):
        for u, v in nx.edge_dfs(T, r):
            T.nodes[v]['load'] = T.edges[u, v]['load']
            if u == r:
                subtree += 1
            T.nodes[v]['subtree'] = subtree
        rootload = 0
        for nbr in T.neighbors(r):
            rootload += T.nodes[nbr]['load']
        T.nodes[r]['load'] = rootload
    return T
