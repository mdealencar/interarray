# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import pyomo.environ as pyo
import networkx as nx
import numpy as np
import math
from collections import defaultdict
from ..crossings import gateXing_iter, edgeset_edgeXing_iter
from ..interarraylib import G_from_site


# class MILPmaker():

#     def __init__(self, A):
#         pass


def make_MILP_length(A, gateXings_constraint=False, gates_limit=False,
                     branching=True):
    '''
    MILP pyomo model for the collector system length minimization.
    A is the networkx graph with the available edges (gate edges
    should not be included, but will be assumed available).

    gates_limit: if True, use the minimum feasible number of gates
    (total for all roots); if False, no limit is imposed; if a number,
    use it as the limit.

    branching: if True, allow subtrees to branch; if False no branching.
    '''
    M = A.graph['M']
    N = A.number_of_nodes() - M
    d2roots = A.graph['d2roots']

    ## Model definition

    # Create model
    m = pyo.AbstractModel()

    # Sets

    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    # the model uses directed edges (except for gate edges), so a duplicate
    # set of edges is created with the reversed tuples
    E = tuple(((u, v) if u < v else (v, u))
              for u, v in A_nodes.edges())
    Eʹ = tuple((v, u) for u, v in E)

    m.diE = pyo.Set(initialize=E + Eʹ)
    m.N = pyo.RangeSet(0, N - 1)
    m.R = pyo.RangeSet(-M, -1)

    # Parameters
    m.d = pyo.Param(m.diE,
                    domain=pyo.PositiveReals,
                    name='edge_cost',
                    initialize=lambda m, u, v: A.edges[(u, v)]['weight'])

    m.g = pyo.Param(m.R, m.N,
                    domain=pyo.PositiveReals,
                    name='gate_cost',
                    initialize=lambda m, r, n: d2roots[n, r])

    m.k = pyo.Param(domain=pyo.PositiveIntegers,
                    name='capacity')

    # Variables
    m.Be = pyo.Var(m.diE, domain=pyo.Binary, initialize=0)
    m.De = pyo.Var(m.diE, domain=pyo.NonNegativeIntegers,
                   bounds=(0, m.k - 1), initialize=0)

    def init_gates(m, r, n):
        return A.nodes[n]['root'] == r

    m.Bg = pyo.Var(m.R, m.N,
                   domain=pyo.Binary,
                   initialize=init_gates)
    m.Dg = pyo.Var(m.R, m.N,
                   domain=pyo.NonNegativeIntegers,
                   bounds=(0, m.k),
                   initialize=init_gates)

    ## Constraints

    # total number of edges must be equal to number of non-root nodes
    m.cons_edges_eq_nodes = pyo.Constraint(
        rule=lambda m: (sum(m.Be[u, v] for u, v in m.diE)
                        + sum(m.Bg[r, n] for r in m.R for n in m.N) == N)
    )

    # each pair of nodes can only have one of the directed edges between them active
    m.cons_one_diEdge = pyo.Constraint(
        E,
        rule=lambda m, u, v: m.Be[u, v] + m.Be[v, u] <= 1
    )

    # each node is connected to a single root
    m.cons_one_root = pyo.Constraint(
        m.N,
        rule=lambda m, n: sum(m.Bg[:, n]) <= 1)

    # gate-edge crossings
    if gateXings_constraint:
        m.cons_gateXedge = pyo.Constraint(
            gateXing_iter(A),
            rule=lambda m, u, v, r, n: m.Be[u, v] + m.Be[v, u] + m.Bg[r, n] <= 1
        )

    # edge-edge crossings
    doubleXings = []
    tripleXings = []
    for Xing in edgeset_edgeXing_iter(A):
        if len(Xing) == 2:
            doubleXings.append(Xing)
        else:
            tripleXings.append(Xing)

    if doubleXings:
        m.cons_edgeXedge = pyo.Constraint(
            doubleXings,
            rule=lambda m, u, v, s, t:
                m.Be[u, v] + m.Be[v, u] + m.Be[s, t] + m.Be[t, s] <= 1
        )

    if tripleXings:
        m.cons_edgeXedgeXedge = pyo.Constraint(
            tripleXings,
            rule=lambda m, u, v, s, t, w, y:
                m.Be[u, v] + m.Be[s, t] + m.Be[w, y] +
                m.Be[v, u] + m.Be[t, s] + m.Be[y, w] <= 1
        )

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
        m.R, m.N,
        rule=lambda m, r, n: m.Dg[r, n] <= m.k*m.Bg[r, n]
    )
    m.cons_gate_active_iff_demand_ub = pyo.Constraint(
        m.R, m.N,
        rule=lambda m, r, n: m.Bg[r, n] <= m.Dg[r, n]
    )

    # TODO: multiple cable types - THAT INVOLVES CHANGING m.Be
    # code below is a quick draft (does NOT work at all)
    if False:
        m.T = pyo.Set(range_number_of_cable_types)
        m.Bte = pyo.Var(m.E, m.T)
        # constraints
        for edge in E:
            # sums over t in T
            pyo.summation(m.Bte[edge]) + pyo.summation(m.Bte[edgeʹ]) <= 1
        for edge in E + Eʹ:
            # sums over t in T
            pyo.summation(m.Bte[edge]) <= m.De[edge]
            pyo.summation(m.k[t]*m.Bte[edge][t]) >= m.De[edge]

    # flow consevation at each node
    m.cons_flow_conservation = pyo.Constraint(
        m.N,
        rule=lambda m, u: (sum((m.De[u, v] - m.De[v, u])
                               for v in A_nodes.neighbors(u))
                           + sum(m.Dg[r, u] for r in m.R)) == 1
    )

    # gates limit
    if gates_limit:
        rule = lambda m: ((sum(m.Bg[r, u] for r in m.R for u in m.N)
                           == math.ceil(N/m.k)) if isinstance(gates_limit, bool) else
                          (sum(m.Bg[r, u] for r in m.R for u in m.N)
                           <= gates_limit))
        m.gates_limit = pyo.Constraint(rule=rule)

    # non-branching
    if not branching:
        # just need to limit incoming edges since the outgoing are
        # limited by the m.cons_one_out_edge
        m.non_branching = pyo.Constraint(
            m.N,
            rule=lambda m, u: sum(m.Be[v, u] for v in A_nodes.neighbors(u)) <= 1
        )

    # assert all nodes are connected to some root
    m.cons_all_nodes_connected = pyo.Constraint(
        rule=lambda m: sum(m.Dg[r, n] for r in m.R for n in m.N) == N
    )

    # valid inequalities
    m.cons_min_gates_required = pyo.Constraint(
        rule=lambda m: sum(m.Bg[r, n] for r in m.R for n in m.N) >= math.ceil(N/m.k)
    )
    m.cons_incoming_demand_limit = pyo.Constraint(
        m.N,
        rule=lambda m, u: sum(m.De[v, u] for v in A_nodes.neighbors(u)) <= m.k - 1
    )
    m.cons_one_out_edge = pyo.Constraint(
        m.N,
        rule=lambda m, u: (sum(m.Be[u, v] for v in A_nodes.neighbors(u))
                           + sum(m.Bg[r, u] for r in m.R) == 1)
    )

    ## Objective
    m.length = pyo.Objective(
        expr=lambda m: pyo.sum_product(m.d, m.Be) + pyo.sum_product(m.g, m.Bg),
        sense=pyo.minimize,
    )

    # TODO: remove redundancy and make it more uniform wrt ortools.py
    m.site = {k: A.graph[k] for k in ('M', 'VertexC', 'boundary', 'name')}
    m.A = A

    return m


def MILP_solution_to_G(model):
    '''Translate a MILP pyomo solution to a networkx graph'''
    G = G_from_site(model.site)
    M = model.site['M']
    N = G.number_of_nodes() - M
    P = model.A.graph['planar'].copy()
    diagonals = model.A.graph['diagonals']

    # gates
    G.add_weighted_edges_from(
        ((r, n, round(model.Dg[r, n].value))
         for (r, n), bg in model.Bg.items()
         if bg.value > 0.5),
        weight='load'
    )
    # node-node edges
    G.add_weighted_edges_from(
        ((u, v, round(model.De[u, v].value))
         for (u, v), be in model.Be.items()
         if be.value > 0.5),
        weight='load'
    )

    # transfer edge attributes from A to G
    nx.set_edge_attributes(
        G, {(u, v): data
            for u, v, data in model.A.edges(data=True)})
    # if A is not available, use model.d
    # to store edge costs as edge attribute
    # nx.set_edge_attributes(G, model.d, 'length')

    gates_not_in_A = G.graph['gates_not_in_A'] = defaultdict(list)

    # propagate loads from edges to nodes
    subtree = -1
    Subtree = defaultdict(list)
    gnT = np.empty((N,), dtype=int)
    Root = np.empty((N,), dtype=int)
    for r in range(-M, 0):
        nx.set_edge_attributes(G, model.g, 'length')
        for u, v in nx.edge_dfs(G, r):
            G.nodes[v]['load'] = G[u][v]['load']
            if u == r:
                subtree += 1
                gate = v
                # check if gate is not expanded Delaunay
                if v not in model.A[r]:
                    # A may not have some gate edges
                    G[u][v]['length'] = model.g[(u, v)]
                    gates_not_in_A[r].append(v)
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
    G.graph['capacity'] = model.k.value
    G.graph['overfed'] = [len(G[r])/math.ceil(N/model.k.value)*M
                          for r in range(-M, 0)]
    G.graph['edges_created_by'] = 'MILP.pyomo'
    G.graph['has_loads'] = True

    return G
