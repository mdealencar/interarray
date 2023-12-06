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


def make_MILP_length(A, κ, gateXings_constraint=False, gates_limit=False,
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

    m.N = pyo.RangeSet(0, N - 1)
    #  m.R = pyo.RangeSet(-M, -1)

    m.branches = pyo.RangeSet(0, math.ceil(N//κ))

    m.gates = tuple((-1, n) for n in range(N))
    m.pairs = pyo.Set(initialize=(sum((tuple((u, v) for v in range(u + 1, N))
                                      for u in range(N)), ()) + m.gates))

    m.node2pairs_map = {n: ((-1, n),) + tuple((v, n) for v in range(0, n))
                        + tuple((n, u) for u in range(n + 1, N))
                        for n in m.N}

    ##############
    # Parameters #
    ##############

    m.d = pyo.Param(m.E,
                    domain=pyo.PositiveReals,
                    name='edge_cost',
                    initialize=lambda m, u, v: A[u][v]['length'])

    #  m.g = pyo.Param(m.R, m.N,
    m.g = pyo.Param(m.N,
                    domain=pyo.PositiveReals,
                    name='gate_cost',
                    initialize=lambda m, n: d2roots[n, -1])
    #                initialize=lambda m, r, n: d2roots[n, r])

    m.κ = pyo.Param(domain=pyo.PositiveIntegers,
                    name='capacity', default=κ)

    #############
    # Variables #
    #############

    m.Be = pyo.Var(m.E, domain=pyo.Binary, initialize=0)
    #  m.De = pyo.Var(m.diE, domain=pyo.NonNegativeIntegers,
    #                 bounds=(0, m.κ - 1), initialize=0)

    #  m.Bg = pyo.Var(m.R, m.N,
                   #  domain=pyo.Binary,
                   #  initialize=0)
    #  m.Dg = pyo.Var(m.R, m.N,
    #                 domain=pyo.NonNegativeIntegers,
    #                 bounds=(0, m.κ),
    #                 initialize=0)
    # flags branch to which each edge belongs
    #  m.Bb = pyo.Var(m.branches, m.Ne, domain=pyo.Binary, initialize=0)
    # flags node that is the gate of each branch
    #  m.Brooted_by = pyo.Var(m.branches, m.N, domain=pyo.Binary, initialize=0)

    m.Ba = pyo.Var(m.pairs, domain=pyo.Binary, initialize=0)
    m.Bg = pyo.Var(m.N,
                   domain=pyo.Binary,
                   initialize=0)

    ###############
    # Constraints #
    ###############

    # link m.Ba and m.Be
    m.cons_edge_implies_adjacency = pyo.Constraint(
        m.E,
        rule=lambda m, u, v: m.Ba[u, v] >= m.Be[u, v])

    # link m.Ba and m.Bg
    m.cons_gate_implies_adjacency = pyo.Constraint(
        m.gates,
        rule=lambda m, u, v: m.Ba[u, v] >= m.Bg[v])

    # transitivity constraints
    m.cons_transit = pyo.ConstraintList()
    for i in range(-1, N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                m.cons_transit.add(m.Ba[i, j] + m.Ba[j, k] - m.Ba[i, k] <= 1)
                m.cons_transit.add(m.Ba[i, j] - m.Ba[j, k] + m.Ba[i, k] <= 1)
                m.cons_transit.add(-m.Ba[i, j] + m.Ba[j, k] + m.Ba[i, k] <= 1)

    # each node's component has at most κ other nodes
    m.cons_branch_size = pyo.Constraint(
        m.N, rule=lambda m, n: (sum(m.Ba[e] for e in m.node2pairs_map[n])
                                <= m.κ))
    # TODO: if using the minimum number of branches, set a constraint for the
    # minimum subtree size as well
    m.cons_fixnum_gates = pyo.Constraint(
        rule=lambda m: sum(m.Bg.values()) == math.ceil(N/κ))

    m.cons_all_nodes_rooted = pyo.Constraint(
        rule=lambda m: sum(m.Ba[g] for g in m.gates) == N)

    # TODO: the formula is supposed to be N*(κ - 1)//2, but this seems to make
    #       the problem impossible
    #  m.cons_adjacency_minimum = pyo.Constraint(
    #      rule=lambda m: sum(m.Ba.values()) >= (N - N % κ)*(κ - 1)//2)

    # total number of edges must be equal to number of non-root nodes
    m.cons_edges_eq_nodes = pyo.Constraint(
        rule=lambda m: (sum(m.Be.values()) + sum(m.Bg.values()) == N)
    )

    # at most one gate per branch
    #  m.cons_single_gate = pyo.Constraint(
    #      m.pairs,
    #      rule=lambda m, u, v: m.Bg[u] + m.Bg[v] + m.Ba[u, v] <= 2)

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
                                        #  + m.Bg[r, n] <= 1)
                                        + m.Bg[n] <= 1)
        )

    # edge-edge crossings
    def edgeXedge_rule(m, *vertices):
        lhs = sum((m.Be[u, v]
                   if u >= 0 else
                   #  m.Bg[u, v])
                   m.Bg[v])
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
    #      rule=lambda m, u, v: m.De[(u, v)] <= m.κ*m.Be[(u, v)]
    #  )
    #  m.cons_edge_active_iff_demand_ub = pyo.Constraint(
    #      m.diE,
    #      rule=lambda m, u, v: m.Be[(u, v)] <= m.De[(u, v)]
    #  )
    #  m.cons_gate_active_iff_demand_lb = pyo.Constraint(
    #      m.R, m.N,
    #      rule=lambda m, r, n: m.Dg[r, n] <= m.κ*m.Bg[r, n]
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
    #          pyo.summation(m.κ[t]*m.Bte[edge][t]) >= m.De[edge]

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
    #                  == math.ceil(N/m.κ))
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
    #                      >= math.ceil(N/m.κ))
    #  )
    #  m.cons_incoming_demand_limit = pyo.Constraint(
    #      m.N,
    #      rule=lambda m, u: (sum(m.De[v, u] for v in A_nodes.neighbors(u))
    #                         <= m.κ - 1)
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
        expr=lambda m: (pyo.sum_product(m.d, m.Be)
                        + pyo.sum_product(m.g, m.Bg)),
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
        ((-1, n) for n, bg in model.Bg.items()
         if bg.value > 0.5)
    )
    # node-node edges
    G.add_edges_from(
        (e
         for e, be in model.Be.items()
         if be.value > 0.5)
    )

    # set the 'reverse' edge attribute
    # node-node edges
    nx.set_edge_attributes(
        G,
        {(u, v): v > u for (u, v), be in model.Be.items() if be.value > 0.5},
        name='reverse')
    # gate edges
    for r in range(-M, 0):
        for n in G[r]:
            G[r][n]['reverse'] = False

    # transfer edge attributes from A to G
    nx.set_edge_attributes(
        G, {(u, v): data
            for u, v, data in A.edges(data=True)})
    for u, v, edgeD in G.edges(data=True):
        if 'type' in edgeD:
            del edgeD['type']

    gates_not_in_A = G.graph['gates_not_in_A'] = defaultdict(list)

    # propagate loads from edges to nodes
    subtree = -1
    Subtree = defaultdict(list)
    gnT = np.empty((N,), dtype=int)
    Root = np.empty((N,), dtype=int)
    for r in range(-M, 0):
        for u, v in nx.edge_dfs(G, r):
            if 'type' in G[u][v]:
                del G[u][v]['type']
            #  G.nodes[v]['load'] = G[u][v]['load']
            if u == r:
                subtree += 1
                gate = v
                # check if gate is not expanded Delaunay
                if v not in A[r]:
                    # A may not have some gate edges
                    #  G[u][v]['length'] = model.g[(u, v)]
                    G[u][v]['length'] = model.g[v]
                    gates_not_in_A[r].append(v)
            Subtree[gate].append(v)
            G.nodes[v]['subtree'] = subtree
            gnT[v] = gate
            Root[v] = r
            # update the planar embedding to include any Delaunay diagonals
            # used in G; the corresponding crossing Delaunay edge is removed
            u, v = (u, v) if u < v else (v, u)
            s = diagonals.get((u, v))
            if s is not None:
                t = P[u][s]['ccw']  # same as P[v][s]['cw']
                P.add_half_edge_cw(u, v, t)
                P.add_half_edge_cw(v, u, s)
                P.remove_edge(s, t)
        #  rootload = 0
        #  for nbr in G.neighbors(r):
        #      rootload += G.nodes[nbr]['load']
        #  G.nodes[r]['load'] = rootload

    G.graph.update(
        planar=P,
        Subtree=Subtree,
        Root=Root,
        gnT=gnT,
        capacity=model.κ.value,
        overfed=[len(G[r])/math.ceil(N/model.κ.value)*M
                 for r in range(-M, 0)],
        edges_created_by='MILP.pyomo',
        creation_options=model.creation_options,
        #  has_loads=True,
        fun_fingerprint=model.fun_fingerprint,
    )

    return G
