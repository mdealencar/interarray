# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import networkx as nx
from ortools.sat.python import cp_model
import numpy as np
import math
from collections import defaultdict
from ..crossings import gateXing_iter, edgeset_edgeXing_iter


def make_MILP_length(A, k, gateXings_constraint=False, gates_limit=False,
                     branching=True):
    '''
    MILP OR-tools CP model for the collector system optimization.
    A is the networkx graph with the available edges.

    gateXings_constraint: whether to avoid crossing of gate edges.

    gates_limit: if True, use the minimum feasible number of gates.
    (total for all roots); if False, no limit is imposed; if a number,
    use it as the limit.

    branching: if True, allow subtrees to branch; if False, no branching.
    '''
    M = A.graph['M']
    N = A.number_of_nodes() - M
    d2roots = A.graph['d2roots']

    ## Model definition

    # Create model
    m = cp_model.CpModel()

    # Parameters
    # k = m.NewConstant(3)
    # Sets

    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    # the model uses directed edges, so a duplicate set of edges
    # is created with the reversed tuples (except for gate edges)
    E = tuple(((u, v) if u < v else (v, u))
              for u, v in A_nodes.edges())

    G = tuple((r, n) for n in range(N) for r in range(-M, 0))
    w_E = tuple(A.edges[(u, v)]['weight'] for u, v in E)
    w_G = tuple(d2roots[n, r] for r, n in G)

    # Variables
    # Binary edge present
    Be = {e: m.NewBoolVar(f'E_{e}') for e in E}
    # Binary gate present
    Bg = {(r, n): m.NewBoolVar(f'g_{n}') for r, n in G}
    # Integer demand on edges
    De = {e: m.NewIntVar(-k + 1, k - 1, f'D_{e}') for e in E}
    # Integer demand on gates
    Dg = {(r, n): m.NewIntVar(0, k, f'Dg_{(r, n)}') for r, n in G}

    ## Constraints

    # link edges' demand and binary
    for e in E:
        m.Add(De[e] == 0).OnlyEnforceIf(Be[e].Not())
        m.AddLinearExpressionInDomain(
            De[e],
            cp_model.Domain.FromIntervals([[-k + 1, -1], [1, k - 1]])
        ).OnlyEnforceIf(Be[e])

    # link gates' demand and binary
    for n in range(N):
        for r in range(-M, 0):
            m.Add(Dg[r, n] == 0).OnlyEnforceIf(Bg[r, n].Not())
            m.Add(Dg[r, n] > 0).OnlyEnforceIf(Bg[r, n])

    # total number of edges must be equal to number of non-root nodes
    m.Add(sum(Be.values()) + sum(Bg.values()) == N)

    # gate-edge crossings
    if gateXings_constraint:
        for (u, v, r, n) in gateXing_iter(A):
            m.AddBoolOr(Be[u, v].Not(), Bg[r, n].Not())

    # edge-edge crossings
    doubleXings = []
    tripleXings = []
    for Xing in edgeset_edgeXing_iter(A):
        if len(Xing) == 2:
            doubleXings.append(Xing)
        else:
            tripleXings.append(Xing)

    for (u, v), (s, t) in doubleXings:
        m.AddBoolOr(Be[u, v].Not(), Be[s, t].Not())

    for (u, v), (s, t), (w, y) in tripleXings:
        m.AddAtMostOne(Be[u, v], Be[s, t], Be[w, y])

    # flow consevation at each node
    for u in range(N):
        m.Add(sum(De[u, v] if u < v else -De[v, u]
                  for v in A_nodes.neighbors(u))
              + sum(Dg[r, u] for r in range(-M, 0)) == 1)

    # gates limit
    min_gates = math.ceil(N/k)
    if gates_limit:
        if isinstance(gates_limit, bool) or gates_limit == min_gates:
            # fixed number of gates
            m.Add((sum(Bg[r, u] for r in range(-M, 0) for u in range(N))
                   == math.ceil(N/k)))
        else:
            assert min_gates < gates_limit
            # number of gates within range
            m.AddLinearConstraint(
                sum(Bg[r, u] for r in range(-M, 0) for u in range(N)),
                min_gates,
                gates_limit)
    else:
        # valid inequality: number of gates is at least the minimum
        m.Add(min_gates <= sum(Bg[r, n] for r in range(-M, 0) for n in range(N)))

    if not branching:
        # non-branching (limit the nodes' degrees to 2)
        for u in range(N):
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
        upstream = defaultdict(list)
        for u, v in E:
            direct = m.NewBoolVar(f'up_{u, v}')
            reverse = m.NewBoolVar(f'down_{u, v}')
            # channeling
            m.Add(De[u, v] > 0).OnlyEnforceIf(direct)
            m.Add(De[u, v] <= 0).OnlyEnforceIf(direct.Not())
            upstream[u].append(direct)
            m.Add(De[u, v] < 0).OnlyEnforceIf(reverse)
            m.Add(De[u, v] >= 0).OnlyEnforceIf(reverse.Not())
            upstream[v].append(reverse)
        for n in range(N):
            # single root enforcement is encompassed here
            m.AddAtMostOne(*upstream[n], *tuple(Bg[r, n]
                                                for r in range(-M, 0)))

    # assert all nodes are connected to some root
    m.Add(sum(Dg[r, n] for r in range(-M, 0) for n in range(N)) == N)

    # Objective
    m.Minimize(cp_model.LinearExpr.WeightedSum(Be.values(), w_E)
               + cp_model.LinearExpr.WeightedSum(Bg.values(), w_G))

    # save data structure as model attributes
    m.Be, m.Bg, m.De, m.Dg = Be, Bg, De, Dg
    m.k = k
    m.site = {k: A.graph[k] for k in ('M', 'VertexC', 'boundary', 'name')}
    return m


def MILP_solution_to_G(solver, model, A):
    '''Translate a MILP OR-tools solution to a networkx graph.'''
    # the solution is in the solver object not in the model
    G = nx.create_empty_copy(A)
    M = G.graph['M']
    N = G.number_of_nodes() - M
    P = A.graph['planar'].copy()
    diagonals = A.graph['diagonals']

    # gates
    gates_and_loads = tuple((r, n, solver.Value(model.Dg[r, n]))
                            for (r, n), bg in model.Bg.items()
                            if solver.Value(bg))
    G.add_weighted_edges_from(gates_and_loads, weight='load')
    # node-node edges
    G.add_weighted_edges_from(
        ((u, v, abs(solver.Value(model.De[u, v])))
         for (u, v), be in model.Be.items()
         if solver.Value(be)),
        weight='load'
    )
    nx.set_edge_attributes(G, {(u, v): data
                               for u, v, data in A.edges(data=True)})

    # take care of gates that were not in A
    gates_not_in_A = G.graph['gates_not_in_A'] = defaultdict(list)
    d2roots = A.graph['d2roots']
    for r, n, _ in gates_and_loads:
        if n not in A[r]:
            gates_not_in_A[r].append(n)
            edgeD = G[n][r]
            edgeD['length'] = edgeD['weight'] = d2roots[n, r]

    # propagate loads from edges to nodes
    subtree = -1
    Subtree = defaultdict(list)
    gnT = np.empty((N,), dtype=int)
    Root = np.empty((N,), dtype=int)
    for r in range(-M, 0):
        for u, v in nx.edge_dfs(G, r):
            G.nodes[v]['load'] = G.edges[u, v]['load']
            if u == r:
                subtree += 1
                gate = v
            Subtree[gate].append(v)
            G.nodes[v]['subtree'] = subtree
            gnT[v] = gate
            Root[v] = r
            # update the planar embedding to include any Delaunay diagonals used in G
            # the corresponding crossing Delaunay edge is removed
            u, v = (u, v) if u < v else (v, u)
            s = diagonals.get((u, v))
            if s is not None:
                t = P[u][s]['ccw']  # same as P[v][s]['cw']
                P.add_half_edge_cw(u, v, t)
                P.add_half_edge_cw(v, u, s)
                P.remove_edge(s, t)
    
    G.graph['planar'] = P
    G.graph['Subtree'] = Subtree
    G.graph['Root'] = Root
    G.graph['gnT'] = gnT
    G.graph['capacity'] = model.k
    G.graph['overfed'] = [len(G[r])/math.ceil(N/model.k)*M
                          for r in range(-M, 0)]
    G.graph['edges_created_by'] = 'MILP.ortools'
    G.graph['has_loads'] = True

    return G
