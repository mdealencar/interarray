# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
from collections import defaultdict

import networkx as nx
import numpy as np

from ortools.sat.python import cp_model

from .core import Optimizer
from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..geometric import delaunay
from ..interarraylib import (G_from_site, calcload, fun_fingerprint,
                             remove_detours)
from ..utils import NodeTagger

F = NodeTagger()


def make_MILP_length(A, κ, gateXings_constraint=False, gates_limit=True,
                     branching=True):
    '''
    MILP OR-tools CP model for the collector system optimization.
    A is the networkx graph with the available edges.

    `κ`: cable capacity

    `gateXings_constraint`: whether to avoid crossing of gate edges.

    `gates_limit`: if True, use the minimum feasible number of gates.
    (total for all roots); if a number, use it as the limit.

    `branching`: if True, allow subtrees to branch; if False, no branching.
    '''
    if not gates_limit:
        print('This implementation only works with gates_limit, '
              'forcing it to True')
    M = A.graph['M']
    V = A.number_of_nodes()
    N = V - M

    d2roots = A.graph['d2roots']

    # Prepare data from A
    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    E = tuple(((u, v) if u < v else (v, u))
              for u, v in A_nodes.edges())
    Eʹ = tuple((v, u) for u, v in E)

    diE = E + Eʹ
    G_ = tuple(tuple((r, n) for n in range(N)) for r in range(-M, 0))
    G = sum(G_, ())
    w_E = tuple(A[u][v]['length'] for u, v in E)
    w_G = tuple(d2roots[n, r] for r, n in G)

    # number of tours, aka gates_limit
    τ = math.ceil(N/κ)
    # Begin model definition
    m = cp_model.CpModel()

    # Parameters
    # k = m.NewConstant(3)

    #############
    # Variables #
    #############

    # Binary node belongs to tour t
    Bn = [[m.NewBoolVar(f'N_{F[n]}_{t}') for n in range(N)]
          for t in range(τ)]
    # Binary root belongs to tour t
    Br = [[m.NewBoolVar(f'R_{F[r]}_{t}') for r in range(-M, 0)]
          for t in range(τ)]
    # Binary edge belongs to tour t
    Be = [{(u, v): m.NewBoolVar(f'E_{F[u]}-{F[v]}_{t}')
          for u, v in diE} for t in range(τ)]
    #  Be = {e: m.NewBoolVar(f'E_{F[e[0]]}-{F[e[1]]}') for e in diE}
    # Binary gate belongs to tour t
    Bg = [{(r, n): m.NewBoolVar(f'G_{F[r]}-{F[n]}_{t}')
           for (r, n) in G} for t in range(τ)]
    # Binary gate belongs to any tour
    #  Bg = {r: {n: m.NewBoolVar(f'G_{F[r]}-{F[n]}')
    #            for r, n in root_gates}
    #        for r, root_gates in enumerate(G_, start=-M)}
    # These are for the return arcs from nodes to hub (no arc cost)
    #  Bnodeshub = {n: m.NewBoolVar(f'H_{F[n]}')
    #               for n in range(N)}

    ###############
    # Constraints #
    ###############

    # prepare the part of circuit identical in all branches
    #  if M == 1:
    #      # only one root, make it the hub
    #      hub = V - 1
    #      base_circuit = []
    #  else:
    #      # more than one root, create a hub
    #      hub = V
    #      #  Bhubroots = {r: m.NewBoolVar(f'H_{F[r]}')
    #      #               for r, nodes in range(-M, 0)}
    #      #  Bhubroots = {k: {r: m.NewBoolVar(f'H_{F[r]}_{k}')
    #      #                   for r, nodes in range(-M, 0)}
    #      #               for k in range(τ)}
    #      # arcs from return hub to roots
    #      #  base_circuit = [(V, V + ρ, var) for ρ, var in Bhubroots.items()]
    #  # arcs from nodes to return hub
    #  base_circuit += [(n, hub, var) for n, var in Bnodeshub.items()]
    #  # arcs from roots to nodes
    #  base_circuit += [(V + ρ, n, var) for ρ, gates in Bg.items()
    #                   for n, var in gates.items()]
    #  # arcs between nodes (both ways)
    #  base_circuit += [(u, v, var) for (u, v), var in Be.items()]
    # circuits (one per branch)
    for t, (tBn, tBr, tBe, tBg) in enumerate(zip(Bn, Br, Be, Bg)):
        #  circuit = base_circuit.copy()
        circuit = []
        if M > 1:
            hub = V
            # arcs from return hub to roots (auxiliary variables created)
            circuit += [(hub, V + r, m.NewBoolVar(f'H_{F[r]}_{t}'))
                        for r in range(-M, 0)]
        else:
            hub = V - 1
        # arcs from nodes to return hub (auxiliary variables created)
        circuit += [(n, hub, m.NewBoolVar(f'H_{F[n]}_{t}')) for n in range(N)]
        # arcs from roots to nodes
        circuit += [(V + r, n, var) for (r, n), var in tBg.items()]
        # arcs between nodes (both ways)
        circuit += [(u, v, var) for (u, v), var in tBe.items()]
        # looping arcs prevent node n from being included in the path
        # i.e. if the node is not assigned to circuit k, neither are its edges
        circuit += [(n, n, var.Not()) for n, var in enumerate(tBn)]
        circuit += [(ρ, ρ, var.Not())
                    for ρ, var in enumerate(tBr, start=V - M)]
        m.AddCircuit(circuit)

    # each node is in exactly one tour
    for n in range(N):
        m.AddExactlyOne(*(Bn[t][n] for t in range(τ)))

    # each tour has exactly one root
    for tBr in Br:
        m.AddExactlyOne(*tBr)

    # each tour has at most κ nodes
    for tBn in Bn:
        m.Add(sum(tBn) <= κ)

    #  # gate-edge crossings
    #  if gateXings_constraint:
    #      for e, (r, n) in gateXing_iter(A):
    #          m.AddAtMostOne(Be[e], Bg[r][n])
    #
    #  # edge-edge crossings
    #  for Xing in edgeset_edgeXing_iter(A):
    #      m.AddAtMostOne(Be[u, v] if u >= 0 else Bg[u][v]
    #                     for u, v in Xing)

    # assert all nodes are connected to some root (using gate edge demands)
    #  m.Add(sum(Dg[r, n] for r in range(-M, 0) for n in range(N)) == N)

    #############
    # Objective #
    #############

    # 2*w_E is tuple concatenation, not multiplication
    #  m.Minimize(cp_model.LinearExpr.WeightedSum(Be.values(), 2*w_E)
    m.Minimize(sum((cp_model.LinearExpr.WeightedSum(Be[t].values(), 2*w_E)
                   + cp_model.LinearExpr.WeightedSum(Bg[t].values(), w_G))
                   for t in range(τ)))

    # save data structure as model attributes
    m.Be, m.Bg, m.Bn, m.Br = Be, Bg, Bn, Br
    m.k = κ
    m.site = {key: A.graph[key]
              for key in ('M', 'VertexC', 'boundary', 'name')}
    m.creation_options = dict(gateXings_constraint=gateXings_constraint,
                              gates_limit=gates_limit,
                              branching=branching)
    m.fun_fingerprint = fun_fingerprint()
    return m


def MILP_warmstart_from_G(m: cp_model.CpModel, G: nx.Graph):
    '''
    Only implemented for non-branching models.
    '''
    if not G.graph.get('has_loads'):
        calcload(G)
    if G.graph.get('D', 0) > 0:
        G = remove_detours(G)
    m.ClearHints()
    upstream = getattr(m, 'upstream', None)
    if upstream is not None:
        let_branch = True
    for (u, v), Be in m.Be.items():
        is_in_G = (u, v) in G.edges
        m.AddHint(Be, is_in_G)
        De = m.De[u, v]
        if is_in_G:
            edgeD = G.edges[u, v]
            m.AddHint(De, edgeD['load']*(1 if edgeD['reverse'] else -1))
        else:
            m.AddHint(De, 0)
    for rn, Bg in m.Bg.items():
        is_in_G = rn in G.edges
        m.AddHint(Bg, is_in_G)
        Dg = m.Dg[rn]
        m.AddHint(Dg, G.edges[rn]['load'] if is_in_G else 0)


def MILP_solution_to_G(model, solver, A=None):
    '''Translate a MILP OR-tools solution to a networkx graph.'''
    # the solution is in the solver object not in the model
    if A is None:
        G = G_from_site(model.site)
        A = delaunay(G)
    else:
        G = nx.create_empty_copy(A)
    M = G.graph['M']
    N = G.number_of_nodes() - M
    P = A.graph['planar'].copy()
    diagonals = A.graph['diagonals']
    G.add_nodes_from(range(-M, 0), type='oss')
    G.add_nodes_from(range(N), type='wtg')

    # gates and edges
    gates = []
    edges = []
    for tBg, tBe in zip(model.Bg, model.Be):
        gates.append(next((r, n)
                          for (r, n), var in tBg.items()
                          if solver.Value(var)))
        edges.append(tuple((u, v)
                           for (u, v), var in tBe.items()
                           if solver.Value(var)))
    edges_ = sum(edges, ())
    G.add_edges_from(edges_)
    G.add_edges_from(gates, reverse=False)

    # set the 'reverse' edge attribute
    # node-node edges
    nx.set_edge_attributes(
        G, {(u, v): (v < u) for u, v in edges_}, name='reverse')

    # transfer edge attributes from A to G
    nx.set_edge_attributes(G, {(u, v): data
                               for u, v, data in A.edges(data=True)})
    for u, v, edgeD in G.edges(data=True):
        if 'type' in edgeD:
            del edgeD['type']

    # take care of gates that were not in A
    gates_not_in_A = G.graph['gates_not_in_A'] = defaultdict(list)
    d2roots = A.graph['d2roots']
    for r, n in gates:
        if n not in A[r]:
            gates_not_in_A[r].append(n)
            G[n][r]['length'] = d2roots[n, r]

    # propagate loads from edges to nodes
    #  subtree = -1
    #  Subtree = defaultdict(list)
    #  gnT = np.empty((N,), dtype=int)
    #  Root = np.empty((N,), dtype=int)
    #  for r in range(-M, 0):
    #      for u, v in nx.edge_dfs(G, r):
    #          if 'type' in G[u][v]:
    #              del G[u][v]['type']
    #          G.nodes[v]['load'] = G.edges[u, v]['load']
    #          if u == r:
    #              subtree += 1
    #              gate = v
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
        capacity=model.k,
        overfed=[len(G[r])/math.ceil(N/model.k)*M
                 for r in range(-M, 0)],
        edges_created_by='MILP.ortools_circuit',
        creation_options=model.creation_options,
        #  has_loads=True,
        has_loads=False,
        fun_fingerprint=model.fun_fingerprint,
    )

    return G
