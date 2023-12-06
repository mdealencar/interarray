# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
from collections import defaultdict

import networkx as nx
import numpy as np

import pyomo.environ as pyo

from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..geometric import delaunay
from ..interarraylib import G_from_site, fun_fingerprint, calcload


def make_MILP_length(A, k, gateXings_constraint=False, gates_limit=False,
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

    # Create model
    m = pyo.ConcreteModel()

    # Sets

    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    # the model uses directed edges (except for gate edges), so a duplicate
    # set of edges is created with the reversed tuples
    m.E = tuple(((u, v) if u < v else (v, u))
                for u, v in A_nodes.edges())
    m.Ne = pyo.RangeSet(0, len(m.E) - 1)
    #  Eʹ = tuple((v, u) for u, v in E)

    #  m.diE = pyo.Set(initialize=E + Eʹ)
    m.N = pyo.RangeSet(0, N - 1)
    #  m.R = pyo.RangeSet(-M, -1)

    m.branches = pyo.RangeSet(0, math.ceil(N//k))

    ##############
    # Parameters #
    ##############

    #  m.d = pyo.Param(m.diE,
    m.d = pyo.Param(m.Ne,
                    domain=pyo.PositiveReals,
                    name='edge_cost',
                    initialize=lambda m, e: A.edges[m.E[e]]['length'])

    #  m.g = pyo.Param(m.R, m.N,
    m.g = pyo.Param(m.N,
                    domain=pyo.PositiveReals,
                    name='gate_cost',
                    initialize=lambda m, n: d2roots[n, -1])
    #                initialize=lambda m, r, n: d2roots[n, r])

    m.k = pyo.Param(domain=pyo.PositiveIntegers,
                    name='capacity', default=k)

    #############
    # Variables #
    #############

    #  m.Be = pyo.Var(m.E, domain=pyo.Binary, initialize=0)
    #  m.De = pyo.Var(m.diE, domain=pyo.NonNegativeIntegers,
    #                 bounds=(0, m.k - 1), initialize=0)

    #  m.Bg = pyo.Var(m.R, m.N,
                   #  domain=pyo.Binary,
                   #  initialize=0)
    #  m.Dg = pyo.Var(m.R, m.N,
    #                 domain=pyo.NonNegativeIntegers,
    #                 bounds=(0, m.k),
    #                 initialize=0)
    # flags branch to which each edge belongs
    m.Bb = pyo.Var(m.branches, m.Ne, domain=pyo.Binary, initialize=0)
    # flags node that is the gate of each branch
    m.Brooted_by = pyo.Var(m.branches, m.N, domain=pyo.Binary, initialize=0)

    ###############
    # Constraints #
    ###############

    # each branch has at most k - 1 edges
    m.cons_subtree_size = pyo.Constraint(
        m.branches, rule=lambda m, b: sum(m.Bb[b, e] for e in m.Ne) <= k - 1)
    # TODO: if using the minimum number of branches, set a constraint for the
    # minimum subtree size as well

    # each edge belongs to at most one branch
    m.cons_edges_in_single_subtree = pyo.Constraint(
        m.Ne, rule=lambda m, e: sum(m.Bb[b, e] for b in m.branches) <= 1)

    # all edges with a common node belong to the same branch
    m.cons_node_edges_same_branch_src = pyo.Constraint(
        m.Ne, m.branches,
        rule=lambda m, e, b: (m.Bb[b, e]
                              + sum(m.Bb[bb, ee]
                                    for ee in A_nodes.edges(m.E[e][0])
                                    for bb in m.branches if bb != b) <= 1)
    )
    # total number of edges must be equal to number of non-root nodes
    m.cons_edges_eq_nodes = pyo.Constraint(
        rule=lambda m: (sum(m.Bb.values())  #[b, e] for b in m.branches for e in m.Ne)
                        + sum(m.Brooted_by.values())  # [b, n] for b in m.branches for n in m.N)
                        == N)
    )

    # exactly one gate per branch
    # TODO: fix this constraint to account for empty branches
    m.cons_branches_rooted = pyo.Constraint(
        m.branches,
        rule=lambda m, b: sum(m.Brooted_by[b, n] for n in m.N) == 1)
    # node roots a branch iif one of its edges belongs to the branch
    m.cons_gate_in_branch = pyo.Constraint(
        m.N, m.branches,
        rule=lambda m, n, b: (sum(m.Bb[b, m.E.index((u, v) if u < v else (v, u))]
                                  for (u, v) in A_nodes.edges(n))
                              >= m.Brooted_by[b, n]))

    # if edge ⟨u, v⟩ is active, then u and v belong to the same branch
    #  m.cons_same_branch = pyo.Constraint(
    #      #  m.E,
    #      m.E, m.branches,
    #      rule=lambda m, u, v, b: (
    #          2*m.Be[u, v] <= sum(m.Bb[bb, u] + m.Bb[bb, v]
    #                              for bb in m.branches if bb != b)))
    #          #  m.Bb[b, u] + m.Bb[b, v]
    #          #  >= 2*m.Be[u, v] - sum(m.Bb[bb, u] + m.Bb[bb, v]
    #          #                        for bb in m.branches if bb != b)))
    #      #  rule=lambda m, u, v: sum((m.Bb[b, u] - m.Bb[b, v])*b for b in m.branches) != 1 - m.Be[u, v])
    #      #  rule=lambda m, u, v, b: m.Bb[b, u] + 2*m.Bb[b, v] == 3*m.Be[u, v])

    # enforce a single directed edge between each node pair
    #  m.cons_one_diEdge = pyo.Constraint(
    #      E,
    #      rule=lambda m, u, v: m.Be[u, v] + m.Be[v, u] <= 1
    #  )

    # each node is connected to a single root
    #  m.cons_one_root = pyo.Constraint(
    #      m.N,
    #      rule=lambda m, n: sum(m.Bg[:, n]) <= 1)

    # gate-edge crossings
    if gateXings_constraint:
        m.cons_gateXedge = pyo.Constraint(
            gateXing_iter(A),
            rule=lambda m, u, v, r, n: (m.Be[u, v]
                                        + m.Bg[r, n] <= 1)
        )

    # edge-edge crossings
    def edgeXedge_rule(m, *vertices):
        lhs = sum((sum(m.Bb[b, m.E.index((u, v))] for b in m.branches)
                   if u >= 0 else
                   sum(m.Brooted_by[b, v] for b in m.branches))
                  for u, v in zip(vertices[::2],
                                  vertices[1::2]))
        return lhs <= 1
    doubleXings = []
    tripleXings = []
    for Xing in edgeset_edgeXing_iter(A):
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
    #  m.cons_edge_active_iff_demand_lb = pyo.Constraint(
    #      m.diE,
    #      rule=lambda m, u, v: m.De[(u, v)] <= m.k*m.Be[(u, v)]
    #  )
    #  m.cons_edge_active_iff_demand_ub = pyo.Constraint(
    #      m.diE,
    #      rule=lambda m, u, v: m.Be[(u, v)] <= m.De[(u, v)]
    #  )
    #  m.cons_gate_active_iff_demand_lb = pyo.Constraint(
    #      m.R, m.N,
    #      rule=lambda m, r, n: m.Dg[r, n] <= m.k*m.Bg[r, n]
    #  )
    #  m.cons_gate_active_iff_demand_ub = pyo.Constraint(
    #      m.R, m.N,
    #      rule=lambda m, r, n: m.Bg[r, n] <= m.Dg[r, n]
    #  )

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

    # flow consevation at each node
    #  m.cons_flow_conservation = pyo.Constraint(
    #      m.N,
    #      rule=lambda m, u: (sum((m.De[u, v] - m.De[v, u])
    #                             for v in A_nodes.neighbors(u))
    #                         + sum(m.Dg[r, u] for r in m.R)) == 1
    #  )

    # gates limit
    #  if gates_limit:
    #      def gates_limit_eq_rule(m):
    #          return (sum(m.Bg[r, u] for r in m.R for u in m.N)
    #                  == math.ceil(N/m.k))
    #
    #      def gates_limit_ub_rule(m):
    #          return (sum(m.Bg[r, u] for r in m.R for u in m.N)
    #                  <= gates_limit)
    #
    #      m.gates_limit = pyo.Constraint(rule=(gates_limit_eq_rule
    #                                           if isinstance(gates_limit, bool)
    #                                           else gates_limit_ub_rule))
    #
    # non-branching
    if not branching:
        # just need to limit incoming edges since the outgoing are
        # limited by the m.cons_one_out_edge
        m.non_branching = pyo.Constraint(
            m.N,
            rule=lambda m, u: (sum(m.Be[v, u] for v in A_nodes.neighbors(u))
                               <= 1)
        )

    # assert all nodes are connected to some root
    #  m.cons_all_nodes_connected = pyo.Constraint(
    #      rule=lambda m: sum(m.Dg[r, n] for r in m.R for n in m.N) == N
    #  )

    # valid inequalities
    #  m.cons_min_gates_required = pyo.Constraint(
    #      rule=lambda m: (sum(m.Bg[r, n] for r in m.R for n in m.N)
    #                      >= math.ceil(N/m.k))
    #  )
    #  m.cons_incoming_demand_limit = pyo.Constraint(
    #      m.N,
    #      rule=lambda m, u: (sum(m.De[v, u] for v in A_nodes.neighbors(u))
    #                         <= m.k - 1)
    #  )
    #  m.cons_one_out_edge = pyo.Constraint(
    #      m.N,
    #      rule=lambda m, u: (sum(m.Be[u, v] for v in A_nodes.neighbors(u))
    #                         + sum(m.Bg[r, u] for r in m.R) == 1)
    #  )

    #############
    # Objective #
    #############

    m.length = pyo.Objective(
        expr=lambda m: (sum(m.d[e]*sum(m.Bb[b, e] for b in m.branches)
                            for e in m.Ne)
                        + sum(sum(m.g[n]*m.Brooted_by[b, n] for n in m.N)
                              for b in m.branches)),
        sense=pyo.minimize,
    )

    m.creation_options = dict(gateXings_constraint=gateXings_constraint,
                              gates_limit=gates_limit,
                              branching=branching)
    m.site = {key: A.graph[key]
              for key in ('M', 'VertexC', 'boundary', 'name')}
    m.fun_fingerprint = fun_fingerprint()
    return m


def MILP_warmstart_from_G(m: pyo.ConcreteModel, G: nx.Graph):
    Ne = len(m.diE)//2
    N = len(m.N)
    # the first half of diE has all the edges with u < v
    for u, v in list(m.diE)[:Ne]:
        if (u, v) in G.edges:
            if G[u][v]['reverse']:
                m.Be[u, v] = 1
                m.De[u, v] = G[u][v]['load']
            else:
                m.Be[v, u] = 1
                m.De[v, u] = G[u][v]['load']
    for r in m.R:
        nbr = list(G.neighbors(r))
        for n in nbr:
            ref = r
            # first skip any detour nodes
            while n >= N:
                a, b = G.neighbors(n)
                c = a if b == ref else b
                ref = n
                n = c
            m.Bg[r, n] = 1
            m.Dg[r, n] = G[n][ref]['load']


def MILP_solution_to_G(model, solver=None, A=None):
    '''Translate a MILP pyomo solution to a networkx graph.'''
    if A is None:
        G = G_from_site(model.site)
        A = delaunay(G)
        P = A.graph['planar']
    else:
        G = nx.create_empty_copy(A)
        P = A.graph['planar'].copy()
    M = model.site['M']
    N = G.number_of_nodes() - M
    diagonals = A.graph['diagonals']
    G.add_nodes_from(range(-M, 0), type='oss')
    G.add_nodes_from(range(N), type='wtg')

    # gates
    G.add_edges_from(
        ((-1, n) for (b, n), val in model.Brooted_by.items()
         if val.value > 0.5)
    )
    # node-node edges
    G.add_edges_from(
        (model.E[e]
         for (b, e), bb in model.Bb.items()
         if bb.value > 0.5)
    )

    # set the 'reverse' edge attribute
    # node-node edges
    #  nx.set_edge_attributes(
    #      G,
    #      {(u, v): v > u for (u, v), be in model.Be.items() if be.value > 0.5},
    #      name='reverse')
    # gate edges
    #  for r in range(-M, 0):
    #      for n in G[r]:
    #          G[r][n]['reverse'] = False

    # transfer edge attributes from A to G
    nx.set_edge_attributes(
        G, {(u, v): data
            for u, v, data in A.edges(data=True)})

    gates_not_in_A = G.graph['gates_not_in_A'] = defaultdict(list)

    #  calcload(G)
    # propagate loads from edges to nodes
    #  subtree = -1
    #  Subtree = defaultdict(list)
    #  gnT = np.empty((N,), dtype=int)
    #  Root = np.empty((N,), dtype=int)
    #  for r in range(-M, 0):
    #      for u, v in nx.edge_dfs(G, r):
    #          if 'type' in G[u][v]:
    #              del G[u][v]['type']
    #          G.nodes[v]['load'] = G[u][v]['load']
    #          if u == r:
    #              subtree += 1
    #              gate = v
    #              # check if gate is not expanded Delaunay
    #              if v not in A[r]:
    #                  # A may not have some gate edges
    #                  G[u][v]['length'] = model.g[(u, v)]
    #                  gates_not_in_A[r].append(v)
    #          Subtree[gate].append(v)
    #          G.nodes[v]['subtree'] = subtree
    #          gnT[v] = gate
    #          Root[v] = r
    #          # update the planar embedding to include any Delaunay diagonals
    #          # used in G; the corresponding crossing Delaunay edge is removed
    #          u, v = (u, v) if u < v else (v, u)
    #          s = diagonals.get((u, v))
    #          if s is not None:
    #              t = P[u][s]['ccw']  # same as P[v][s]['cw']
    #              P.add_half_edge_cw(u, v, t)
    #              P.add_half_edge_cw(v, u, s)
    #              P.remove_edge(s, t)
    #      rootload = 0
    #      for nbr in G.neighbors(r):
    #          rootload += G.nodes[nbr]['load']
    #      G.nodes[r]['load'] = rootload

    G.graph.update(
        planar=P,
        #  Subtree=Subtree,
        #  Root=Root,
        #  gnT=gnT,
        capacity=model.k.value,
        overfed=[len(G[r])/math.ceil(N/model.k.value)*M
                 for r in range(-M, 0)],
        edges_created_by='MILP.pyomo_partition',
        creation_options=model.creation_options,
        has_loads=False,
        fun_fingerprint=model.fun_fingerprint,
    )

    return G
