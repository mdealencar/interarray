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


class ORtoolsOptimizer(Optimizer):

    def make_model(self, k: int) -> cp_model.CpModel:
        A, A_nodes = self.A, self.A_nodes
        E, G = self.E, self.G
        N, M = self.N, self.M
        let_branch = self.options['let_branch']
        let_cross = self.options['let_cross']
        limit_gates = self.options['limit_gates']

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
        if limit_gates:
            if isinstance(limit_gates, bool) or limit_gates == min_gates:
                # fixed number of gates
                m.Add((sum(Bg[r, u] for r in range(-M, 0) for u in range(N))
                       == math.ceil(N/k)))
                min_gate_load = N % k
            else:
                assert min_gates < limit_gates, (
                        f'Infeasible: N/k > gates_limit (N = {N}, k = {k},'
                        f' gates_limit = {limit_gates}).')
                # number of gates within range
                m.AddLinearConstraint(
                    sum(Bg[r, u] for r in range(-M, 0) for u in range(N)),
                    min_gates,
                    limit_gates)
        else:
            # valid inequality: number of gates is at least the minimum
            m.Add(min_gates <= sum(Bg[r, n]
                                   for r in range(-M, 0)
                                   for n in range(N)))

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
                m.Add(Dg[r, n] >= min_gate_load).OnlyEnforceIf(Bg[r, n])

        # total number of edges must be equal to number of non-root nodes
        m.Add(sum(Be.values()) + sum(Bg.values()) == N)

        # gate-edge crossings
        if not let_cross:
            for e, g in gateXing_iter(A):
                m.AddAtMostOne(Be[e], Bg[g])

        # edge-edge crossings
        for Xing in edgeset_edgeXing_iter(A):
            m.AddAtMostOne(*(Be[u, v] if u >= 0 else Bg[u, v]
                             for u, v in Xing))

        # flow consevation at each node
        for u in range(N):
            m.Add(sum(De[u, v] if u < v else -De[v, u]
                      for v in A_nodes.neighbors(u))
                  + sum(Dg[r, u] for r in range(-M, 0)) == 1)

        if not let_branch:
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
            for n in range(N):
                # single root enforcement is encompassed here
                m.AddAtMostOne(
                    *upstream[n].values(), *tuple(Bg[r, n] for r in range(-M, 0))
                )
            m.upstream = upstream

        # assert all nodes are connected to some root (using gate edge demands)
        m.Add(sum(Dg[r, n] for r in range(-M, 0) for n in range(N)) == N)

        #############
        # Objective #
        #############

        m.Minimize(cp_model.LinearExpr.WeightedSum(Be.values(), self.w_E)
                   + cp_model.LinearExpr.WeightedSum(Bg.values(), self.w_G))

        # save data structure as model attributes
        m.Be, m.Bg, m.De, m.Dg = Be, Bg, De, Dg
        m.k = k
        #  m.site = {key: A.graph[key]
        #            for key in ('M', 'VertexC', 'boundary', 'name')}
        m.options = self.options
        #  m.creation_options = dict(gateXings_constraint=gateXings_constraint,
        #                            gates_limit=gates_limit,
        #                            branching=branching)
        m.fun_fingerprint = fun_fingerprint()
        self.model = m

    def warmstart(self, G):
        G = super().warmstart(G)

    def run(self):
        solver = self.solver
        m = self.model
        self.solver.solve(m, **self.solver_options)
