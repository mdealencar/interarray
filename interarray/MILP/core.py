# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

from abc import ABCMeta, abstractmethod

import networkx as nx

from ..geometric import complete_graph, delaunay
from ..interarraylib import calcload, remove_detours


class Optimizer(metaclass=ABCMeta):

    def __init__(self, G, let_branch=True, let_cross=True,
                 limit_gates=False, delaunay_based=True):

        self.options = dict(let_branch=let_branch, let_cross=let_cross,
                            limit_gates=limit_gates,
                            delaunay_based=delaunay_based)
        A = delaunay(G) if delaunay_based else complete_graph(G)
        self.A = A

        M = A.graph['M']
        N = A.number_of_nodes() - M
        self.N, self.M = N, M
        d2roots = A.graph['d2roots']

        # Prepare data from A
        A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
        self.A_nodes = A_nodes
        E = tuple(((u, v) if u < v else (v, u))
                  for u, v in A_nodes.edges())
        G = tuple((r, n) for n in range(N) for r in range(-M, 0))
        w_E = tuple(A[u][v]['length'] for u, v in E)
        w_G = tuple(d2roots[n, r] for r, n in G)

        self.__dict__.update(A_nodes=A_nodes, E=E, G=G, w_E=w_E, w_G=w_G)

    @abstractmethod
    def make_model(self, k: int):
        pass

    @abstractmethod
    def warmstart(self, G):
        if getattr(self, 'm', None) is None:
            print('No model created, run `make_model()` first.')
            return
        if not G.graph.get('has_loads'):
            calcload(G)
        if G.graph.get('D', 0) > 0:
            G = remove_detours(G)
        return G

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def G_from_solution(self):
        pass
