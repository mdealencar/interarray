# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import heapq
import math
from collections import defaultdict, deque, namedtuple
from itertools import chain

import matplotlib
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from loguru import logger

from .crossings import gateXing_iter
from .mesh import planar_flipped_by_routeset
from .geometric import rotate, rotation_checkers_factory
from .interarraylib import bfs_subtree_loads
from .utils import NodeStr, NodeTagger
from .plotting import gplot, scaffolded

trace, debug, info, success, warn, error, critical = (
    logger.trace, logger.debug, logger.info, logger.success,
    logger.warning, logger.error, logger.critical)

F = NodeTagger()

NULL = np.iinfo(int).min
PseudoNode = namedtuple('PseudoNode', 'node sector parent dist d_hop'.split())


class PathNodes(dict):
    '''Helper class to build a tree that uses clones of prime nodes
    (i.e. where the same prime node can appear as more that one node).'''

    def __init__(self):
        super().__init__()
        self.count = 0
        self.prime_from_id = {}
        self.ids_from_prime_sector = defaultdict(list)
        self.last_added = None

    def add(self, _source, sector, parent, dist, d_hop):
        if parent not in self:
            error('attempted to add an edge in `PathNodes` to nonexistent'
                  'parent ({})', parent)
        _parent = self.prime_from_id[parent]
        for prev_id in self.ids_from_prime_sector[_source, sector]:
            if self[prev_id].parent == parent:
                self.last_added = prev_id
                return prev_id
        id = self.count
        self.count += 1
        self[id] = PseudoNode(_source, sector, parent, dist, d_hop)
        self.ids_from_prime_sector[_source, sector].append(id)
        self.prime_from_id[id] = _source
        trace('pseudoedge «{}->{}» added', F[_source], F[_parent])
        self.last_added = id
        return id


class PathFinder():
    '''
    Router for gates that don't belong to the PlanarEmbedding of the graph.
    Initialize it with a detour-free routeset `G` and it will find paths from
    all nodes to the nearest root without crossing any used edges.

    Only edges in graph attribute 'tentative' or, lacking that, edges with the
    attribute 'kind' == 'tentative' are checked for crossings.

    These paths can be used to replace the existing gates that cross other
    edges by gate paths with detours.

    All of `G`'s edges must be in `planar` (may need to call
    `planar_flipped_by_routeset()` beforehand).

    Example:
    ========

    H = PathFinder(G, planar=P, A=A).create_detours()
    '''

    def __init__(self, Gʹ: nx.Graph, *,
                 planar: nx.PlanarEmbedding,
                 A: nx.Graph | None = None,
                 branching: bool = True) -> None:
        G = Gʹ.copy()
        M, N, B = (G.graph[k] for k in 'MNB')
        C = G.graph.get('C', 0)
        assert not G.graph.get('D'), 'Gʹ has already has detours.'

        # Block for facilitating the printing of debug messages.
        allnodes = np.arange(N + M + B + 3)
        allnodes[-M:] = range(-M, 0)
        self.n2s = NodeStr(allnodes, N + B + 3)

        info('BEGIN pathfinding on "{}" (#wtg = {})',
             G.graph.get('name') or G.graph.get('handle') or 'unnamed', N)

        # tentative will be copied later, by initializing a set from it.
        tentative = G.graph.get('tentative')
        hooks2check = []
        if tentative is None:
            # TODO: this case should be removed ('tentative' attr mandatory)
            tentative = []
            for r in range(-M, 0):
                gates = set(n for n in G.neighbors(r)
                            if G[r][n].get('kind') == 'tentative')
                tentative.extend((r, n) for n in gates)
                hooks2check.append(gates)
        else:
            hooks2check.extend(set() for _ in range(M))
            for r, n in tentative:
                hooks2check[r].add(n)

        Xings = list(
            gateXing_iter(
                G, hooks=[np.fromiter(h2c, count=len(h2c), dtype=int)
                          for h2c in hooks2check],
                borders=planar.graph.get('constraint_edges')))

        self.G, self.Xings, self.tentative = G, Xings, set(tentative)
        if not Xings:
            # no crossings, there is no point in pathfinding
            info('Graph has no crossings, skipping path-finding.')
            return

        # clone2prime must be a copy of the one from Gʹ
        clone2prime = list(G.graph.get('clone2prime', ()))
        # TODO: work around PathFinder getting metrics for the supertriangle
        #       nodes -> do away with A metrics, eliminate A from args
        if A is None:
            VertexC = G.graph['VertexC']
            if G.graph.get('is_normalized'):
                supertriangleC = G.graph['norm_scale']*(
                    planar.graph['supertriangleC'] - G.graph['norm_offset'])
            VertexC = np.vstack((VertexC[:N + B],
                                 supertriangleC,
                                 VertexC[-M:]))
            d2roots = cdist(VertexC[:-M], VertexC[-M:])
            Rank = None
            diagonals = None
        else:
            VertexC = A.graph['VertexC']
            d2roots = A.graph['d2roots']
            Rank = A.graph.get('d2rootsRank')
            diagonals = A.graph['diagonals']
        self.saved_shortened_contours = saved_shortened_contours = []
        shortened_contours = G.graph.get('shortened_contours')
        if shortened_contours is not None:
            # G has edges that shortcut some longer paths along P edges.
            # We need to put these paths back in G to do P edge flips.
            # The changes made here are undone in `create_detours()`.
            clone2prime = G.graph['clone2prime']
            clone_offset = N + B
            for (s, t), (midpath, shortpath) in shortened_contours.items():
                # G follows shortpath, but we want it to follow midpath
                subtree_id = G.nodes[t]['subtree']
                stored_edges = []
                path = [s] + shortpath + [t]
                for u_, v_ in zip(path[:-1], path[1:]):
                    u = (u_ if u_ < N else
                         clone_offset + np.flatnonzero(clone2prime == u_)[-1])
                    v = (v_ if v_ < N else
                         clone_offset + np.flatnonzero(clone2prime == v_)[-1])
                    stored_edges.append((u, v, G[u][v]))
                    # the nodes are left for later reuse
                    G.remove_edge(u, v)
                helper_edges = []
                u = s
                for v in midpath:
                    # this will use border nodes, watchout!
                    helper_edges.append((u, v))
                    G.add_edge(u, v, kind='contour')
                    G.nodes[v]['subtree'] = subtree_id
                    u = v
                helper_edges.append((u, t))
                G.add_edge(u, t, kind='contour')
                saved_shortened_contours.append((stored_edges, helper_edges))
        P = planar_flipped_by_routeset(G, planar=planar, VertexC=VertexC,
                                       diagonals=diagonals)
        self.d2roots = d2roots
        self.d2rootsRank = Rank or rankdata(d2roots, method='dense', axis=0)
        self.predetour_length = Gʹ.size(weight='length')
        creator = G.graph.get('creator')
        if creator is not None and creator[:5] == 'MILP.':
            self.branching = G.graph['method_options']['branching']
        else:
            self.branching = branching

        self.M, self.N, self.B, self.C = M, N, B, C
        self.P, self.VertexC, self.clone2prime = P, VertexC, clone2prime
        self.hooks2check = hooks2check
        self._find_paths()

    def get_best_path(self, n: int):
        '''
        `_.get_best_path(«node»)` produces a `tuple(path, dists)`.
        `path` contains a sequence of nodes from the original
        networx.Graph `G`, from «node» to the closest root.
        `dists` contains the lengths of the segments defined by `paths`.
        '''
        paths = self.paths
        paths_available = tuple((paths[id].dist, id)
                                for id in self.I_path[n].values())
        if paths_available:
            dist, id = min(paths_available)
            path = [n]
            dists = []
            pseudonode = paths[id]
            while id >= 0:
                dists.append(pseudonode.d_hop)
                id = pseudonode.parent
                path.append(paths.prime_from_id[id])
                pseudonode = paths[id]
            return path, dists
        else:
            info('Path not found for «{}»', F[n])
            return [], []

    def _get_sector(self, _node: int, portal: tuple[int, int]):
        '''
        Given a `_node` and a `portal` to which `_node` belongs, the sector is
        the first neighbor of `_node` rotating in the counterclockwise
        direction from the opposite node in `portal` that forms one of G's
        edges with `_node`.
        The sector is a way of identifying from which side of a boundary the
        path is reaching `_node`.
        '''
        # TODO: there is probably a better way to avoid spinning around _node
        if _node >= self.N:
            # _node is in a border (which means it must only be reachable from
            # one side, so that sector becomes irrelevant)
            return NULL
        is_gate = any(_node in Gate for Gate in self.hooks2check)
        _node_degree = self.G.degree[_node]
        if is_gate and _node_degree == 1:
            # special case where a branch with 1 node uses a non_embed gate
            root = self.G.nodes[_node]['root']
            return (NULL
                    if _node == portal[0] else
                    root)
            #  return self.G.nodes[_node]['root']

        _opposite = portal[0] if _node == portal[1] else portal[1]
        _nbr = self.P[_node][_opposite]['ccw']
        # this `while` loop would in some cases loop forever
        #  while ((_node, _nbr) not in self.G.edges):
        #      _nbr = self.P[_node][_nbr]['ccw']
        for _ in range(_node_degree):
            if (_node, _nbr) in self.G.edges:
                break
            _nbr = self.P[_node][_nbr]['ccw']
        return _nbr

    def _rate_wait_add(self, portal: tuple[int, int], _new: int, _apex: int,
                       apex: int):
        I_path = self.I_path
        paths = self.paths
        d_hop = np.hypot(*(self.VertexC[_apex] - self.VertexC[_new]).T)
        pseudoapex = paths[apex]
        d_new = pseudoapex.dist + d_hop
        new_sector = self._get_sector(_new, portal)
        incumbent = I_path[_new].get(new_sector)
        is_better = incumbent is None or d_new < paths[incumbent].dist
        yield d_new, portal, (_new, _apex), is_better
        new = self.paths.add(_new, new_sector, apex, d_new, d_hop)
        self.uncharted[portal] = max(self.uncharted[portal] - 1, 0)
        #  self.uncharted[portal[1], portal[0]] = False
        # get incumbent again, as the situation may have changed
        incumbent = I_path[_new].get(new_sector)
        if incumbent is None or d_new < paths[incumbent].dist:
            self.I_path[_new][new_sector] = new
            debug('{} added with d_path = {:.2f}',
                  self.n2s(_new, _apex), d_new)

    def _advance_portal(self, left: int, right: int):
        G, P, N, B = self.G, self.P, self.N, self.B
        while True:
            # look for children portals
            n = P[left][right]['ccw']
            if n not in P[right] or P[left][n]['ccw'] == right or n < 0:
                # (u, v, n) not a new triangle
                return
            # examine the other two sides of the triangle
            next_portals = []
            for (s, t, side) in ((left, n, 1), (n, right, 0)):
                st_sorted = (s, t) if s < t else (t, s)
                trace('evaluating {}', self.n2s(s, t))
                if (st_sorted not in self.portal_set
                        or (s < N and t < N
                            and G.nodes[s]['subtree'] ==
                            G.nodes[t]['subtree'])):
                    # (s, t) is in G or is bounded by a subtree
                    continue
                next_portals.append(((s, t), side))
            try:
                # this `pop()` will raise IndexError if we are at a dead-end
                first, fside = next_portals.pop()
                # use this instead of the if-else-block when done debugging
                #  yield left, right, (
                #          self._portal_iter(*next_portals[0])
                #          if next_portals
                #          else None)
                if next_portals:
                    second, sside = next_portals[0]
                    debug(f'branching {self.n2s(*first)} and '
                          f'{self.n2s(*second)}')
                    yield (first, fside,
                           chain(((second, sside, None),),
                                 self._advance_portal(*second)))
                else:
                    trace('{}', self.n2s(*first))
                    yield first, fside, None
            except IndexError:
                # dead-end reached
                trace('dead-end: {}–{}', F[left], F[right])
                return
            left, right = first

    def _traverse_channel(self, _apex: int, apex: int, _funnel: list[int],
                          wedge_end: list[int], portal_iter: iter):
        # variable naming notation:
        # for variables that represent a node, they may occur in two versions:
        #     - _node: the index it contains maps to a coordinate in VertexC
        #     - node: contais a pseudonode index (i.e. an index in self.paths)
        #             translation: _node = paths.prime_from_id[node]
        cw, ccw = rotation_checkers_factory(self.VertexC)
        paths = self.paths

        # for next_left, next_right, new_portal_iter in portal_iter:
        for portal, side, new_portal_iter in portal_iter:
            #  print('[tra]')
            if new_portal_iter is not None:
                # spawn a branched traverser
                #  print(f'new channel {self.n2s(_apex, *_funnel)} -> '
                #        f"{F[_new]} {'RIGHT' if side else 'LEFT '}")
                branched_traverser = self._traverse_channel(
                        _apex, apex, _funnel.copy(), wedge_end.copy(),
                        new_portal_iter)
                self.bifurcation = branched_traverser

            _new = portal[side]
            _nearside = _funnel[side]
            _farside = _funnel[not side]
            test = ccw if side else cw

            #  if _nearside == _apex:  # debug info
            #      print(f"{'RIGHT' if side else 'LEFT '} "
            #            f'nearside({F[_nearside]}) == apex({F[_apex]})')
            trace(f"{'RIGHT' if side else 'LEFT '} "
                  f'new({F[_new]}) '
                  f'nearside({F[_nearside]}) '
                  f'farside({F[_farside]}) '
                  f'apex({F[_apex]}), wedge ends: '
                  f'{F[paths.prime_from_id[wedge_end[0]]]}, '
                  f'{F[paths.prime_from_id[wedge_end[1]]]}')
            if _nearside == _apex or test(_nearside, _new, _apex):
                # not infranear
                if test(_farside, _new, _apex):
                    # ultrafar (new overlaps farside)
                    trace('ultrafar')
                    current_wapex = wedge_end[not side]
                    _current_wapex = paths.prime_from_id[current_wapex]
                    _funnel[not side] = _current_wapex
                    contender_wapex = paths[current_wapex].parent
                    _contender_wapex = paths.prime_from_id[contender_wapex]
                    #  print(f"{'RIGHT' if side else 'LEFT '} "
                    #        f'current_wapex({F[_current_wapex]}) '
                    #        f'contender_wapex({F[_contender_wapex]})')
                    while (_current_wapex != _farside
                           and _contender_wapex >= 0
                           and test(_new, _current_wapex, _contender_wapex)):
                        _funnel[not side] = _current_wapex
                        #  wedge_end[not side] = current_wapex
                        current_wapex = contender_wapex
                        _current_wapex = _contender_wapex
                        contender_wapex = paths[current_wapex].parent
                        _contender_wapex = paths.prime_from_id[contender_wapex]
                        #  print(f"{'RIGHT' if side else 'LEFT '} "
                        #        f'current_wapex({F[_current_wapex]}) '
                        #        f'contender_wapex({F[_contender_wapex]})')
                    _apex = _current_wapex
                    apex = current_wapex
                else:
                    trace('inside')
                yield from self._rate_wait_add(portal, _new, _apex, apex)
                wedge_end[side] = paths.last_added
                _funnel[side] = _new
            else:
                # infranear
                trace('infranear')
                current_wapex = wedge_end[side]
                _current_wapex = paths.prime_from_id[current_wapex]
                #  print(f'{F[_current_wapex]}')
                contender_wapex = paths[current_wapex].parent
                _contender_wapex = paths.prime_from_id[contender_wapex]
                while (_current_wapex != _nearside
                       and _contender_wapex >= 0
                       and test(_current_wapex, _new, _contender_wapex)):
                    current_wapex = contender_wapex
                    _current_wapex = _contender_wapex
                    #  print(f'{F[current_wapex]}')
                    contender_wapex = paths[current_wapex].parent
                    _contender_wapex = paths.prime_from_id[contender_wapex]
                yield from self._rate_wait_add(
                    portal, _new, _current_wapex, current_wapex)
                wedge_end[side] = paths.last_added

    def _find_paths(self):
        #  print('[exp] starting _explore()')
        G, P, M, N, B, C = self.G, self.P, self.M, self.N, self.B, self.C
        d2roots = self.d2roots
        d2rootsRank = self.d2rootsRank
        prioqueue = []
        # `uncharted` records whether portals have been traversed
        # (it is orientation-sensitive – two permutations)
        uncharted = defaultdict(lambda: 2)
        paths = self.paths = PathNodes()
        self.uncharted = uncharted
        self.bifurcation = None
        I_path = defaultdict(dict)
        self.I_path = I_path

        # set of portals (i.e. edges of P that are not used in G)
        fnT = G.graph.get('fnT')
        if fnT is not None:
            edges_G_primed = {((u, v) if u < v else (v, u))
                              for u, v in (fnT[edge,] for edge in G.edges)}
            nodes_G_primed = {fnT[n] for n in G.nodes}
        else:
            edges_G_primed = {((u, v) if u < v else (v, u))
                              for u, v in G.edges}
            nodes_G_primed = set(G.nodes)
        ST = N + B
        edges_P = {((u, v) if u < v else (v, u))
                   for u, v in P.edges if u < ST or v < ST}
        portal_set = edges_P - edges_G_primed
        self.portal_set = portal_set
        constraint_edges = P.graph['constraint_edges']

        # launch channel traversers around the roots to the prioqueue
        for r in range(-M, 0):
            paths[r] = PseudoNode(r, r, None, 0., 0.)
            paths.prime_from_id[r] = r
            paths.ids_from_prime_sector[r, r] = [r]
            for left in P.neighbors(r):
                right = P[r][left]['cw']
                portal = (left, right)
                portal_sorted = (right, left) if right < left else portal
                if (right not in P[r] or portal_sorted not in portal_set):
                    # (left, right, root) not a triangle
                    # or (left, right) is not a portal
                    continue
                # flag initial portals as visited
                self.uncharted[portal] = 0
                self.uncharted[right, left] = 0

                if left >= ST or (left in G.nodes and G.degree[left] == 0):
                    sec_left = NULL
                else:
                    sec_left = right
                    while True:
                        sec_left = P[left][sec_left]['ccw']
                        incr_edge = ((sec_left, left) if sec_left < left
                                     else (left, sec_left))
                        if (incr_edge in edges_G_primed
                                or incr_edge in constraint_edges):
                            break
                    if sec_left == r:
                        sec_left = NULL

                d_left, d_right = d2roots[left, r], d2roots[right, r]
                # add the first pseudo-nodes to paths
                wedge_end = [paths.add(left, sec_left, r, d_left, d_left),
                             paths.add(right, r, r, d_right, d_right)]

                # shortest paths for roots' P.neighbors is a straight line
                I_path[left][sec_left], I_path[right][r] = wedge_end

                # prioritize by distance to the closest node of the portal
                closest, d_closest = (
                    (left, d_left)
                    if d2rootsRank[left, r] <= d2rootsRank[right, r]
                    else (right, d_right)
                )
                heapq.heappush(prioqueue, (
                    d_closest, portal, (closest, r), 0,
                    self._traverse_channel(r, r, [left, right], wedge_end,
                                           self._advance_portal(left, right))
                ))
        # TODO: this is arbitrary, should be documented somewhere (or removed)
        MAX_ITER = 10000
        #  MAX_ITER = 300
        # process edges in the prioqueue
        counter = 0
        #  print(f'[exp] starting main loop, |prioqueue| = {len(prioqueue)}')
        while len(prioqueue) > 0 and counter < MAX_ITER:
            # safeguard against infinite loop
            counter += 1
            #  print(f'[exp] {counter}')

            if self.bifurcation is None:
                # no bifurcation, pop the best traverser from the prioqueue
                _d_contender, _portal, _hop, _, traverser = \
                        heapq.heappop(prioqueue)
            else:
                # the last processed portal bifurcated
                # add it to the queue and get the best traverser

                # make the traverser advance one portal
                d_contender, portal, hop, is_better = next(self.bifurcation)
                if is_better or uncharted[portal]:
                    #  print(f'[exp]^pushing dist = {d_contender:.0f}, '
                    #        f'{self.n2s(*hop)} ')
                    _d_contender, _portal, _hop, _, traverser = (
                        heapq.heappushpop(prioqueue, (d_contender, portal,
                                                      hop, counter,
                                                      self.bifurcation)))
                #  else:
                #      print(f'[exp]^traverser {self.n2s(*hop)} was '
                #            'dropped (no better than previous traverser).')
                self.bifurcation = None
            #  print(f'[exp]_popped dist = {_d_contender:.0f}, '
            #        f'{self.n2s(*_hop)} ')
            try:
                # make the traverser advance one portal
                d_contender, portal, hop, is_better = next(traverser)
            except StopIteration:
                #  print(f'[exp]_traverser {self.n2s(*hop)} was '
                #        'dropped (dead-end).')
                pass
            else:
                if is_better or uncharted[portal]:
                    #  print(f'[exp]_pushing dist = {d_contender:.0f}, '
                    #        f'{self.n2s(*hop)} ')
                    heapq.heappush(prioqueue,
                                   (d_contender, portal, hop, counter,
                                    traverser))
                #  else:
                #      print(f'[exp]_traverser {self.n2s(*hop)} was '
                #            'dropped (no better that previous traverser).')
        if counter == MAX_ITER:
            warn('Path-finding loop aborted at MAX_ITER!')
        info('Path-finding loops performed: {}.', counter)

    def _apply_all_best_paths(self, G: nx.Graph):
        '''
        Update G with the paths found by `_find_paths()`.
        '''
        get_best_path = self.get_best_path
        for n in range(self.N):
            for id in self.I_path[n].values():
                if id < 0:
                    # n is a root's neighbor
                    continue
            path, dists = get_best_path(n)
            nx.add_path(G, path, kind='virtual')

    def plot_best_paths(self, ax: matplotlib.axes.Axes | None = None, **kwargs,
                        ) -> matplotlib.axes.Axes:
        '''
        Plot the subtrees of G (without gate edges) overlaid by the shortest
        paths for all nodes.
        '''
        K = nx.subgraph_view(self.G,
                             filter_edge=lambda u, v: u >= 0 and v >= 0)
        ax = gplot(K, ax=ax, **{'infobox': False, 'node_tag': None} | kwargs)
        J = nx.Graph()
        J.add_nodes_from(self.G.nodes)
        self._apply_all_best_paths(J)
        landscape_angle = self.G.graph.get('landscape_angle')
        if landscape_angle:
            VertexC = rotate(self.VertexC, landscape_angle)
        else:
            VertexC = self.VertexC
        nx.draw_networkx_edges(J, pos=VertexC, edge_color='y',
                               alpha=0.5, ax=ax, label='path to root')

        return ax

    def plot_scaffolded(self, ax: matplotlib.axes.Axes | None = None, **kwargs,
                        ) -> matplotlib.axes.Axes:
        '''
        Plot the PlanarEmbedding of G, overlaid by the edges of G that coincide
        with it.
        '''
        return gplot(scaffolded(self.G, P=self.P), ax=ax,
                     **{'infobox': False} | kwargs)

    def create_detours(self):
        '''Reroute all gate edges in G with crossings using detour paths.

        Returns:
            New networkx.Graph (shallow copy of G, with detours).
        '''
        # TODO: create_detours() cannot be called twice. Enforce that!
        G, Xings, tentative = self.G, self.Xings, self.tentative.copy()

        if not Xings:
            for r, n in tentative:
                # remove the 'tentative' kind
                if 'kind' in G[r][n]:
                    del G[r][n]['kind']
            if 'tentative' in G.graph:
                del G.graph['tentative']
            return G

        if self.saved_shortened_contours is not None:
            # Restore shortcut contours as they were before finding paths.
            for stored_edges, helper_edges in self.saved_shortened_contours:
                G.remove_edges_from(helper_edges)
                G.add_edges_from(stored_edges)

        M, N, B, C = self.M, self.N, self.B, self.C
        clone2prime = self.clone2prime.copy()
        paths, I_path = self.paths, self.I_path
        clone_idx = N + B + C
        failed_detours = []

        subtree_from_subtree_id = defaultdict(list)
        subtree_id_from_n = {}
        for n in chain(range(N), range(N + B, clone_idx)):
            subtree_id = G.nodes[n]['subtree']
            subtree_from_subtree_id[subtree_id].append(n)
            subtree_id_from_n[n] = subtree_id

        for r, n in set(gate for _, gate in Xings):
            tentative.remove((r, n))
            subtree_id = subtree_id_from_n[n]
            subtree = subtree_from_subtree_id[subtree_id]
            subtree_load = G.nodes[n]['load']
            # set of nodes to examine is different depending on `branching`
            hookchoices = ([n for n in subtree if n < N]
                           if self.branching else
                           [n, next(h for h in subtree if G.degree[h] == 1)])
            debug('hookchoices: {}', self.n2s(*hookchoices))

            path_options = list(chain.from_iterable(
                ((paths[id].dist, id, hook, sec)
                 for sec, id in I_path[hook].items())
                for hook in hookchoices))
            if not path_options:
                error('subtree of node {} has no non-crossing paths to any '
                      'root: leaving gate as-is', F[n])
                # unable to fix this crossing
                failed_detours.append((r, n))
                continue
            dist, id, hook, sect = min(path_options)
            debug('best: hook = {}, sector = {}, dist = {:.1f}',
                  F[hook], F[sect], dist)

            path = [hook]
            dists = []
            pseudonode = paths[id]
            while id >= 0:
                dists.append(pseudonode.d_hop)
                id = pseudonode.parent
                path.append(paths.prime_from_id[id])
                pseudonode = paths[id]
            if not math.isclose(sum(dists), dist):
                error(f'distance sum ({sum(dists):.1f}) != '
                      f'best distance ({dist:.1f}), hook = {F[hook]}, '
                      f'path: {self.n2s(*path)}')

            debug('path: {}', self.n2s(*path))
            if len(path) < 2:
                error('no path found for {}-{}', F[r], F[n])
                continue
            added_clones = len(path) - 2
            Clone = list(range(clone_idx, clone_idx + added_clones))
            clone_idx += added_clones
            clone2prime.extend(path[1:-1])
            G.add_nodes_from(((c, {'label': F[c],
                                   'kind': 'detour',
                                   'subtree': subtree_id,
                                   'load': subtree_load})
                              for c in Clone))
            if [n, r] != path:
                # TODO: adapt this for contoured gates
                #       maybe that's the place to prune contour clones
                G.remove_edge(r, n)
                if r != path[-1]:
                    debug(f'root changed from {F[r]} to '
                          f'{F[path[-1]]} for subtree of gate {F[n]}, now '
                          f'hooked to {F[path[0]]}')
                G.add_weighted_edges_from(
                    zip(path[:1] + Clone, Clone + path[-1:], dists),
                    weight='length')
                for _, _, edgeD in G.edges(Clone, data=True):
                    edgeD.update(kind='detour', reverse=True)
                if added_clones > 0:
                    # an edge reaching root always has target < source
                    G[Clone[-1]][path[-1]]['reverse'] = False
            else:
                del G[n][r]['kind']
                debug(f'gate {F[n]}–{F[r]} touches a '
                      'node (touched node does not become a detour).')
            if n != path[0]:
                # the hook changed: update 'load' attributes of edges/nodes
                debug('hook changed from {} to {}: recalculating loads',
                      F[n], F[path[0]])

                for node in subtree:
                    del G.nodes[node]['load']

                if Clone:
                    parent = Clone[0]
                    ref_load = subtree_load
                    G.nodes[parent]['load'] = 0
                else:
                    parent = path[-1]
                    ref_load = G.nodes[parent]['load']
                    G.nodes[parent]['load'] = ref_load - subtree_load
                total_parent_load = bfs_subtree_loads(G, parent, [path[0]],
                                                      subtree_id)
                assert total_parent_load == ref_load, \
                    f'detour {F[n]}–{F[path[0]]}: load calculated ' \
                    f'({total_parent_load}) != expected load ({ref_load})'

        # former tentative gates that were not in Xings cease to be tentative
        for r, n in tentative:
            del G[r][n]['kind']

        if failed_detours:
            warn('Failed: ', ' '.join(f'{F[u]}–{F[v]}'
                                      for u, v in failed_detours))
            G.graph['tentative'] = failed_detours
        else:
            del G.graph['tentative']

        fnT = np.arange(M + clone_idx)
        fnT[N + B: clone_idx] = clone2prime
        fnT[-M:] = range(-M, 0)
        G.graph.update(D=clone_idx - N - B - C, fnT=fnT)
        G.graph['detextra'] = G.size(weight='length')/self.predetour_length - 1
        # TODO: there might be some lost contour clones that could be prunned
        return G
