# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import heapq
import math
from collections import defaultdict, deque, namedtuple
from itertools import chain
from typing import Callable, Optional, Tuple

import matplotlib
import networkx as nx
import numpy as np
from loguru import logger

from .crossings import gateXing_iter
from .geometric import planar_over_layout, rotate
from .interarraylib import bfs_subtree_loads
from .utils import NodeStr, NodeTagger
from .plotting import gplot, scaffolded

trace, debug, info, success, warn, error, critical = (
    logger.trace, logger.debug, logger.info, logger.success,
    logger.warning, logger.error, logger.critical)

F = NodeTagger()

NULL = np.iinfo(int).min
PseudoNode = namedtuple('PseudoNode', 'node sector parent dist d_hop'.split())


def rotation_checkers_factory(VertexC: np.ndarray) -> Tuple[
        Callable[[int, int, int], bool],
        Callable[[int, int, int], bool]]:

    def cw(A: int, B: int, C: int) -> bool:
        """return
            True: if A->B->C traverses the triangle ABC clockwise
            False: otherwise"""
        Ax, Ay = VertexC[A]
        Bx, By = VertexC[B]
        Cx, Cy = VertexC[C]
        return (Bx - Ax) * (Cy - Ay) < (By - Ay) * (Cx - Ax)

    def ccw(A: int, B: int, C: int) -> bool:
        """return
            True: if A->B->C traverses the triangle ABC counter-clockwise
            False: otherwise"""
        Ax, Ay = VertexC[B]
        Bx, By = VertexC[A]
        Cx, Cy = VertexC[C]
        return (Bx - Ax) * (Cy - Ay) < (By - Ay) * (Cx - Ax)

    return cw, ccw

class PathNodes(dict):
    '''Helper class to build a tree that uses clones of base nodes
    (i.e. where the same base node can appear as more that one node).'''

    def __init__(self):
        super().__init__()
        self.count = 0
        self.base_from_id = {}
        self.ids_from_base_sector = defaultdict(list)
        self.last_added = None

    def add(self, _source, sector, parent, dist, d_hop):
        if parent not in self:
            error('attempted to add an edge in `PathNodes` to nonexistent'
                  'parent ({})', parent)
        _parent = self.base_from_id[parent]
        for prev_id in self.ids_from_base_sector[_source, sector]:
            if self[prev_id].parent == parent:
                self.last_added = prev_id
                return prev_id
        id = self.count
        self.count += 1
        self[id] = PseudoNode(_source, sector, parent, dist, d_hop)
        self.ids_from_base_sector[_source, sector].append(id)
        self.base_from_id[id] = _source
        debug('pseudoedge «{}->{}» added', F[_source], F[_parent])
        self.last_added = id
        return id


class PathFinder():
    '''
    Router for gates that don't belong to the PlanarEmbedding of the graph.
    Initialize it with a valid layout graph `G` and it will find paths from
    all nodes to the nearest root without crossing any used edges.
    These paths can be used to replace the existing gates that cross other
    edges by gate paths with detours.

    Example:
    ========

    H = PathFinder(G).create_detours()
    '''

    def __init__(self, G, branching=True,
                 only_if_crossings=True):
        M = G.graph['M']
        VertexC = G.graph['VertexC']
        self.VertexC = VertexC
        N = VertexC.shape[0] - M

        # this is just for facilitating printing debug messages
        allnodes = np.arange(N + M)
        allnodes[-M:] = range(-M, 0)
        self.n2s = NodeStr(allnodes, N)

        info('BEGIN pathfinding on "{}" (#wtg = {})', G.graph['name'], N)
        P = G.graph.get('planar') or planar_over_layout(G)
        self.G, self.P, self.M, self.N = G, P, M, N
        # sets of gates (one per root) that are not in the planar embedding
        nonembed_Gates = tuple(
            np.fromiter(set(G.neighbors(r)) - set(P.neighbors(r)), dtype=int)
            for r in range(-M, 0))
        self.nonembed_Gates = nonembed_Gates
        if G.graph['edges_created_by'][:5] == 'MILP.':
            self.branching = G.graph['creation_options']['branching']
        else:
            self.branching = branching
        Xings = list(gateXing_iter(G, gates=nonembed_Gates))
        self.Xings = Xings
        if not Xings and only_if_crossings:
            # no crossings, there is no point in pathfinding
            info('Graph has no crossings, skipping path-finding '
                 '(pass `only_if_crossings=False` to force path-finding).')
            return
        self._find_paths()

    def get_best_path(self, n):
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
                path.append(paths.base_from_id[id])
                pseudonode = paths[id]
            return path, dists
        else:
            info('Path not found for «{}»', F[n])
            return [], []

    def _get_sector(self, _node, portal):
        # TODO: there is probably a better way to avoid spinning around _node
        is_gate = any(_node in Gate for Gate in self.nonembed_Gates)
        _node_degree = len(self.G._adj[_node])
        if is_gate and _node_degree == 1:
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

    def _rate_wait_add(self, portal, _new, _apex, apex):
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

    def _advance_portal(self, left, right):
        G = self.G
        P = self.P
        while True:
            # look for children portals
            n = P[left][right]['ccw']
            if n not in P[right] or P[left][n]['ccw'] == right or n < 0:
                # (u, v, n) not a new triangle
                # is this is the moment to launch a hull crawler?
                return
            # examine the other two sides of the triangle
            next_portals = []
            for (s, t, side) in ((left, n, 1), (n, right, 0)):
                st_sorted = (s, t) if s < t else (t, s)
                if (st_sorted not in self.portal_set
                        or G.nodes[s]['subtree'] == G.nodes[t]['subtree']):
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
                    trace(f'branching {self.n2s(*first)} and '
                          f'{self.n2s(*second)}')
                    yield (first, fside,
                           chain(((second, sside, None),),
                                 self._advance_portal(*second)))
                else:
                    trace('{}', self.n2s(*first))
                    yield first, fside, None
            except IndexError:
                # dead-end reached
                return
            left, right = first

    def _traverse_channel(self, _apex: int, apex: int, _funnel: list,
                          wedge_end: list, portal_iter: iter):
        # variable naming notation:
        # for variables that represent a node, they may occur in two versions:
        #     - _node: the index it contains maps to a coordinate in VertexC
        #     - node: contais a pseudonode index (i.e. an index in self.paths)
        #             translation: _node = paths.base_from_id[node]
        cw, ccw = rotation_checkers_factory(self.VertexC)
        paths = self.paths

        # for next_left, next_right, new_portal_iter in portal_iter:
        for portal, side, new_portal_iter in portal_iter:
            #  print('[tra]')
            _new = portal[side]
            if new_portal_iter is not None:
                # spawn a branched traverser
                #  print(f'new channel {self.n2s(_apex, *_funnel)} -> '
                #        f"{F[_new]} {'RIGHT' if side else 'LEFT '}")
                branched_traverser = self._traverse_channel(
                        _apex, apex, _funnel.copy(), wedge_end.copy(),
                        new_portal_iter)
                self.bifurcation = branched_traverser

            _nearside = _funnel[side]
            _farside = _funnel[not side]
            test = ccw if side else cw

            #  if _nearside == _apex:  # debug info
            #      print(f"{'RIGHT' if side else 'LEFT '} "
            #            f'nearside({F[_nearside]}) == apex({F[_apex]})')
            debug(f"{'RIGHT' if side else 'LEFT '} "
                  f'new({F[_new]}) '
                  f'nearside({F[_nearside]}) '
                  f'farside({F[_farside]}) '
                  f'apex({F[_apex]}), wedge ends: '
                  f'{F[paths.base_from_id[wedge_end[0]]]}, '
                  f'{F[paths.base_from_id[wedge_end[1]]]}')
            if _nearside == _apex or test(_nearside, _new, _apex):
                # not infranear
                if test(_farside, _new, _apex):
                    # ultrafar (new overlaps farside)
                    debug('ultrafar')
                    current_wapex = wedge_end[not side]
                    _current_wapex = paths.base_from_id[current_wapex]
                    _funnel[not side] = _current_wapex
                    contender_wapex = paths[current_wapex].parent
                    _contender_wapex = paths.base_from_id[contender_wapex]
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
                        _contender_wapex = paths.base_from_id[contender_wapex]
                        #  print(f"{'RIGHT' if side else 'LEFT '} "
                        #        f'current_wapex({F[_current_wapex]}) '
                        #        f'contender_wapex({F[_contender_wapex]})')
                    _apex = _current_wapex
                    apex = current_wapex
                else:
                    debug('inside')
                yield from self._rate_wait_add(portal, _new, _apex, apex)
                wedge_end[side] = paths.last_added
                _funnel[side] = _new
            else:
                # infranear
                debug('infranear')
                current_wapex = wedge_end[side]
                _current_wapex = paths.base_from_id[current_wapex]
                #  print(f'{F[_current_wapex]}')
                contender_wapex = paths[current_wapex].parent
                _contender_wapex = paths.base_from_id[contender_wapex]
                while (_current_wapex != _nearside
                       and _contender_wapex >= 0
                       and test(_current_wapex, _new, _contender_wapex)):
                    current_wapex = contender_wapex
                    _current_wapex = _contender_wapex
                    #  print(f'{F[current_wapex]}')
                    contender_wapex = paths[current_wapex].parent
                    _contender_wapex = paths.base_from_id[contender_wapex]
                yield from self._rate_wait_add(
                    portal, _new, _current_wapex, current_wapex)
                wedge_end[side] = paths.last_added

    def _find_paths(self):
        #  print('[exp] starting _explore()')
        G, P, M = self.G, self.P, self.M
        d2roots = G.graph['d2roots']
        d2rootsRank = G.graph['d2rootsRank']
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
        portal_set = set()
        for i, (u, v) in enumerate(P.to_undirected(as_view=True).edges
                                   - G.edges):
            portal_set.add((u, v) if u < v else (v, u))
        self.portal_set = portal_set

        # launch channel traversers around the roots to the prioqueue
        for r in range(-M, 0):
            paths[r] = PseudoNode(r, r, None, 0., 0.)
            paths.base_from_id[r] = r
            paths.ids_from_base_sector[r, r] = [r]
            for left in P.neighbors(r):
                right = P[r][left]['cw']
                portal = (left, right)
                portal_sorted = (right, left) if right < left else portal
                if not (right in P[r] and portal_sorted in portal_set):
                    # (u, v, r) not a triangle or (u, v) is in G
                    continue
                # flag initial portals as visited
                self.uncharted[portal] = 0
                self.uncharted[right, left] = 0

                sec_left = P[left][right]['ccw']
                while (left, sec_left) not in G.edges:
                    sec_left = P[left][sec_left]['ccw']
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
                #      print(f'[exp]^traverser {self.n2s(*hop)} '
                #            'was dropped (no better than previous traverser).')
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
            warn('_find_paths() ended prematurely!')
        info('_find_path looped {} times', counter)

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
            nx.add_path(G, path, type='virtual')

    def plot_best_paths(self,
                        ax: Optional[matplotlib.axes.Axes] = None
                        ) -> matplotlib.axes.Axes:
        '''
        Plot the subtrees of G (without gate edges) overlaid by the shortest
        paths for all nodes.
        '''
        K = nx.subgraph_view(self.G,
                             filter_edge=lambda u, v: u >= 0 and v >= 0)
        ax = gplot(K, ax=ax)
        J = nx.Graph()
        J.add_nodes_from(self.G.nodes)
        self._apply_all_best_paths(J)
        landscape_angle = self.G.graph.get('landscape_angle')
        if landscape_angle:
            VertexC = rotate(self.VertexC, landscape_angle)
        else:
            VertexC = self.VertexC
        nx.draw_networkx_edges(J, pos=VertexC, edge_color='y',
                               alpha=0.3, ax=ax)
        return ax

    def plot_scaffolded(self,
                        ax: Optional[matplotlib.axes.Axes] = None
                        ) -> matplotlib.axes.Axes:
        '''
        Plot the PlanarEmbedding of G, overlaid by the edges of G that coincide
        with it.
        '''
        return gplot(scaffolded(self.G, P=self.P), ax=ax, infobox=False)

    def create_detours(self, in_place: bool = False):
        '''
        Replace all gate edges in G that cross other edges with detour paths.
        If `in_place`, change the G given to PathFinder and return None,
        else return new nx.Graph (G with detours).
        '''
        if in_place:
            G = self.G
        else:
            G = self.G.copy()
            # this self.H is only for debugging purposes
            self.H = G
        if 'crossings' in G.graph:
            # start by assumming that crossings will be fixed by detours
            del G.graph['crossings']
        M, N = self.M, self.N
        Xings = self.Xings

        if not Xings:
            # no crossings were detected, nothing to do
            return None if in_place else G.copy()

        gates2detour = set(gate for _, gate in Xings)

        clone2prime = []
        D = 0
        subtree_map = G.nodes(data='subtree')
        # if G.graph['edges_created_by'] == 'make_MILP_length':
        if G.graph['edges_created_by'][:5] == 'MILP.':
            branching = G.graph['creation_options']['branching']
        else:
            branching = True
        #  print(f'branching: {branching}')
        paths = self.paths
        I_path = self.I_path
        for r, n in gates2detour:
            subtree_id = G.nodes[n]['subtree']
            subtree_load = G.nodes[n]['load']
            # set of nodes to examine is different depending on `branching`
            hookchoices = (  # all the subtree's nodes
                           [n for n, v in subtree_map if v == subtree_id]
                           if branching else
                           # only each subtree's head and tail
                           [n] + [n for n, v in subtree_map
                                  if v == subtree_id and len(G._adj[n]) == 1])
            debug('hookchoices: {}', self.n2s(*hookchoices))

            path_options = list(chain.from_iterable(
                ((paths[id].dist, id, hook, sec)
                 for sec, id in I_path[hook].items())
                for hook in hookchoices))
            if not path_options:
                error('subtree of node {} has no non-crossing paths to any '
                      'root: leaving gate as-is', F[n])
                # unable to fix this crossing
                crossings = G.graph.get('crossings')
                if crossings is None:
                    G.graph['crossings'] = []
                G.graph['crossings'].append((r, n))
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
                path.append(paths.base_from_id[id])
                pseudonode = paths[id]
            if not math.isclose(sum(dists), dist):
            #  assert math.isclose(sum(dists), dist), \
                error(f'distance sum ({sum(dists):.1f}) != '
                      f'best distance ({dist:.1f}), hook = {F[hook]}, '
                      f'path: {self.n2s(*path)}')

            debug('path: {}', self.n2s(*path))
            if len(path) < 2:
                error('no path found for {}-{}', F[r], F[n])
                continue
            Dinc = len(path) - 2
            Clone = list(range(N + D, N + D + Dinc))
            clone2prime.extend(path[1:-1])
            D += Dinc
            G.add_nodes_from(((c, {'label': F[c],
                                   'type': 'detour',
                                   'subtree': subtree_id,
                                   'load': subtree_load})
                              for c in Clone))
            if [n, r] != path:
                G.remove_edge(r, n)
                if r != path[-1]:
                    debug(f'root changed from {F[r]} to '
                          f'{F[path[-1]]} for subtree of gate {F[n]}, now '
                          f'hooked to {F[path[0]]}')
                G.add_weighted_edges_from(
                    zip(path[:1] + Clone, Clone + path[-1:], dists),
                    weight='length')
                for _, _, edgeD in G.edges(Clone, data=True):
                    edgeD.update(type='detour', reverse=True)
                if Dinc > 0:
                    # an edge reaching root always has target < source
                    del G[Clone[-1]][path[-1]]['reverse']
            else:
                debug(f'gate {F[n]}–{F[r]} touches a '
                      'node (touched node does not become a detour).')
            if n != path[0]:
                # the hook changed: update 'load' attributes of edges/nodes
                debug('hook changed from {} to {}: recalculating loads',
                      F[n], F[path[0]])

                for node in [n for n, v in subtree_map if v == subtree_id]:
                    if 'load' in G.nodes[node]:
                        del G.nodes[node]['load']

                if Clone:
                    parent = Clone[0]
                    ref_load = subtree_load
                    G.nodes[parent]['load'] = 0
                else:
                    parent = path[-1]
                    ref_load = G.nodes[parent]['load']
                    G.nodes[parent]['load'] = ref_load - subtree_load
                total_parent_load = bfs_subtree_loads(G, parent, [path[0]], subtree_id)
                assert total_parent_load == ref_load, \
                    f'detour {F[n]}–{F[path[0]]}: load calculated ' \
                    f'({total_parent_load}) != expected load ({ref_load})'

        fnT = np.arange(N + D + M)
        fnT[N: N + D] = clone2prime
        fnT[-M:] = range(-M, 0)
        G.graph.update(D=D, fnT=fnT)
        return None if in_place else G


# >>> BELLOW THIS POINT LAYS THE DEPRECATED CODE: midpoint based paths >>>
class PathSeeker():
    '''
    Deprecated earlier implementation of `PathFinder`. Uses the midpoints of
    the portals to calculate distances for prioritization. Its reroute_gate()
    method is broken, do not use.

    It still exists for the nice view of paths given by its `plot()` method.
    (Besides, it has a more readable implementation of the funnel algorithm.)
    '''

    def __init__(self, G, branching=True,
                 only_if_crossings=True):
        M = G.graph['M']
        VertexC = G.graph['VertexC']
        self.VertexC = VertexC
        N = VertexC.shape[0] - M

        # this is just for facilitating printing debug messages
        allnodes = np.arange(N + M)
        allnodes[-M:] = range(-M, 0)
        self.n2s = NodeStr(allnodes, N)

        info('BEGIN pathfinding on "{}" (#wtg = {})', G.graph['name'], N)
        P = G.graph.get('planar') or planar_over_layout(G)
        self.G, self.P, self.M, self.N = G, P, M, N
        # sets of gates (one per root) that are not in the planar embedding
        nonembed_Gates = tuple(
            np.fromiter(set(G.neighbors(r)) - set(P.neighbors(r)), dtype=int)
            for r in range(-M, 0))
        self.nonembed_Gates = nonembed_Gates
        if G.graph['edges_created_by'][:5] == 'MILP.':
            self.branching = G.graph['creation_options']['branching']
        else:
            self.branching = branching
        Xings = list(gateXing_iter(G, gates=nonembed_Gates))
        self.Xings = Xings
        if not Xings and only_if_crossings:
            # no crossings, there is no point in pathfinding
            info('Graph has no crossings, skipping path-finding '
                 '(pass `only_if_crossings=False` to force path-finding).')
            return
        self.create()

    def _get_scaffold(self) -> nx.Graph:
        scaffold = getattr(self, 'scaffold', None)
        if scaffold is None:
            scaffold = scaffolded(self.G, P=self.P)
            self.scaffold = scaffold
        return scaffold

    def path_by_funnel(self, s, t, channel):
        '''
        (only used by the legacy midpoint aproach, but can be used
        independently as long as `self.VertexC` has the coordinates)

        Funnel algorithm (aka string pulling):
        parameters:
            `s`: source node
            `t`: target node
            `channel`: sequence of node pairs defining the portals
                       (must be ⟨left, right⟩ wrt direction s -> t)
        returns:
            `path`: sequence of nodes (including `s` and `t`)
        '''
        # original algorithm description in section 4.4 of th MSc thesis:
        # Efficient Triangulation-Based Pathfinding (2007, Douglas Jon Demyen)
        # https://era.library.ualberta.ca/items/3075ea07-5eb5-44e9-b863-ae898bf00fe1

        # also inspired by (C implementations):
        # Ash Hamnett: Funnel Algorithm
        # http://ahamnett.blogspot.com/2012/10/funnel-algorithm.html
        # Simple Stupid Funnel Algorithm
        # http://digestingduck.blogspot.com/2010/03/simple-stupid-funnel-algorithm.html?m=1
        # (but instead of resetting the portal when a new apex is chosen, this
        # implementation keeps track of the nodes that need to be revisited)

        path = [s]
        apex = s
        # Left and Right deques are complicated to explain: they are needed
        # for backtracking after one side remains stuck at a channel boundary
        # violation while the other advances several portals before violating
        # the funnel and triggering a new apex.
        Left = deque()
        Right = deque()
        channel.append((t, t))
        left, right = channel[0]
        cw, ccw = rotation_checkers_factory(self.VertexC)

        counter = 0
        for next_left, next_right in channel[1:]:
            # log = (f'{counter:2d} '
            #        f'⟨{F[left]}–{F[apex]}–{F[right]}⟩ '
            #        + n2s(next_left, next_right))
            counter += 1
            # left side
            if next_left != left:
                # log += f' L:{n2s(next_left)} '
                if left == apex or cw(left, next_left, apex):
                    # log += ' bou:pass '
                    # non-overlapping
                    # test overlap with other side
                    if cw(right, next_left, apex):
                        # overlap on the right
                        # log += ' opo:fail '
                        # other side is the new apex
                        apex = right
                        path.append(apex)
                        while Right:
                            right = Right.popleft()
                            if (cw(right, next_left, apex)
                                and (not Right
                                     or cw(right, Right[0], apex))):
                                apex = right
                                path.append(apex)
                            elif cw(right, next_right, apex):
                                Right.appendleft(right)
                                next_right = right
                                break
                        right = next_right
                    left = next_left
                else:
                    # log += ' bou:fail '
                    if not Left or next_left != Left[-1]:
                        Left.append(next_left)
            # right side
            if next_right != right:
                # log += f' R:{n2s(next_right)} '
                # test overlap with boundary
                if right == apex or cw(next_right, right, apex):
                    # log += ' bou:pass '
                    # non-overlapping
                    # test overlap with other side
                    if cw(next_right, left, apex):
                        # overlap on the left
                        # log += ' opo:fail '
                        # other side is the new apex
                        apex = left
                        path.append(apex)
                        while Left:
                            left = Left.popleft()
                            if (cw(next_right, left, apex)
                                and (not Left
                                     or cw(Left[0], left, apex))):
                                apex = left
                                path.append(apex)
                            elif cw(next_left, left, apex):
                                Left.appendleft(left)
                                next_left = left
                                break
                        left = next_left
                    right = next_right
                else:
                    # log += ' bou:fail '
                    if not Right or next_right != Right[-1]:
                        Right.append(next_right)
            # print(log, f'⟨{F[left]}–{F[apex]}–{F[right]}⟩',
            #       n2s(*path) if path else '')
        if path[-1] != t:
            path.append(t)
        return path

    def plot(self, ax):
        if not getattr(self, 'Gmidpt', False):
            self.create()
        ax = gplot(self._get_scaffold(), node_tag='label', ax=ax)
        nx.draw_networkx_edges(self.Gmidpt, pos=self.PortalC, ax=ax, edge_color='y',
                               arrowsize=3, node_size=0, alpha=0.3)
        return ax

    def create(self):
        G = self.G
        P = self.P
        M = self.M
        VertexC = self.VertexC
        idx_from_portal = {}
        #  N_portals = (P.number_of_edges()//2
        #               - G.number_of_edges()
        #               + sum((len(Gate) for Gate in self.nonembed_Gates)))
        all_portals = P.to_undirected(as_view=True).edges - G.edges
        N_portals = len(all_portals)
        portal_from_idx = np.empty((N_portals, 2), dtype=int)
        for i, (u, v) in enumerate(all_portals):
            u, v = (u, v) if u < v else (v, u)
            portal_from_idx[i] = u, v
            idx_from_portal[u, v] = i
        self.idx_from_portal = idx_from_portal
        self.portal_from_idx = portal_from_idx

        self.Gmidpt = nx.DiGraph()
        self.Gmidpt.add_nodes_from(range(-M, 0), dist=0.)
        PortalC = np.empty((N_portals + M, 2), dtype=float)
        PortalC[:N_portals] = (VertexC[portal_from_idx[:, 0]]
                               + VertexC[portal_from_idx[:, 1]])/2
        PortalC[-M:] = VertexC[-M:]
        self.PortalC = PortalC

        # items in que prioqueue:
        # (distance-to-root from portal following midpoints,
        #  parent idx, portal idx, dist parent-portal,
        #  portal vertex low, portal vertex high,
        #  rotation from low->high for next vertex,
        #  rootsector vertex idx)
        prioqueue = []

        # add edges around the roots to the prioqueue
        for r in range(-M, 0):
            for u in P.neighbors(r):
                v = P[u][r]['cw']
                (u, v), cw = ((u, v), True) if u < v else ((v, u), False)
                if not (v in P[r] and (u, v) in idx_from_portal):
                    # (u, v, r) not a triangle or (u, v) is in G
                    continue
                p = idx_from_portal[(u, v)]
                d_hop = np.hypot(*(PortalC[p] - VertexC[r]).T)
                # for the 1st hop: d_path = d_hop
                heapq.heappush(prioqueue, (d_hop, r, p, d_hop, u, v, cw, p))

        # TODO: this is arbitrary, should be documented somewhere (or removed)
        MAX_ITER = 10000
        # process edges in the prioqueue
        counter = 0
        while len(prioqueue) > 0 and counter < MAX_ITER:
            # safeguard against infinite loop
            counter += 1

            # get next item in the prioqueue
            (dist, parent, p, d_hop,
             u, v, cw, rootsector) = heapq.heappop(prioqueue)
            if p in self.Gmidpt:
                # some other route reached p at a lower cost
                continue
            # rev is true if the portal nodes are ordered (right, left)
            # instead of (left, right)
            self.Gmidpt.add_node(p, dist=dist, rootsector=rootsector, rev=not cw)
            self.Gmidpt.add_edge(p, parent, length=d_hop)

            # look for children portals
            spin = 'cw' if cw else 'ccw'
            n = P[u][v][spin]
            if n not in P[v] or P[u][n][spin] == v or n < 0:
                # (u, v, n) not a new triangle
                continue
            # examine the other two sides of the triangle
            for (s, t) in ((u, n), (n, v)):
                (s, t), next_cw = ((s, t), cw) if s < t else ((t, s), not cw)
                if (s, t) not in idx_from_portal:
                    # (s, t, r) not a triangle or (s, t) is in G
                    continue
                next_p = idx_from_portal[(s, t)]
                next_dist = np.hypot(*(PortalC[next_p] - PortalC[p]).T)
                heapq.heappush(prioqueue,
                               (dist + next_dist, p, next_p, next_dist,
                                s, t, next_cw, rootsector))
        if counter >= MAX_ITER:
            print('pathfinder generation INTERRUPTED!!!'
                  ' reached iterations limit = ', counter)

    def check(self):
        # TODO: check() and _check_recursive() belong to testing code
        for r in range(-self.G.graph['M'], 0):
            for pred in self.Gmidpt.predecessors(r):
                self.check_recursive(
                    pred, self.Gmidpt[pred][r]['length'], pred)

    def check_recursive(self, entry, prev_dist, prev_rootsector):
        # TODO: check() and _check_recursive() belong to testing code
        rootsector = self.Gmidpt.nodes[entry]['rootsector']
        if rootsector != prev_rootsector:
            print('rootsector changed:',
                  self.n2s(*self.portal_from_idx[entry]),
                  'previously: ',
                  self.n2s(*self.portal_from_idx[prev_rootsector]),
                  'now: ', self.n2s(*self.portal_from_idx[rootsector]))
        for pred in self.Gmidpt.predecessors(entry):
            pred_dist = self.Gmidpt.nodes[pred]['dist']
            cummulative = prev_dist + self.Gmidpt[pred][entry]['length']
            diff = cummulative - pred_dist
            if diff != 0:
                print('distance mismatch:',
                      self.n2s(*self.portal_from_idx[entry]),
                      self.n2s(*self.portal_from_idx[pred]),
                      f'diff = {diff:.0f}')
            self._check_recursive(pred, cummulative, rootsector)

    def reroute_gate(self, s):
        '''
        DEPRECATED: this algorithm is defective and will not be fixed

        Return shortest path (sequence of nodes, possibly with detours) to link
        the subtree with gate `s` to a root node (another node from the subtree
        may be chosen as gate).
            :param s: some gate node
        Returns:
            tuple(actual_dist, pred_dist, path, dists)
        '''
        # TODO: It is possible to have edges in self.P which were not
        # included in the pathfinder graph. These are confined between
        # the convex hull and a single subtree. A corner case would be
        # an entire subtree in that region, for which no path can be
        # found (it would require leaving the convex hull).

        G = self.G
        P = self.P
        subtree = G.nodes[s]['subtree']
        # print(F[s], subtree)
        # incumbent = defaultdict(dict)  # by rootsector, then by dist
        subtree_pdict = defaultdict(list)

        # log = ''
        # STEP 1) Get the portals around the nodes of the subtree.
        nodes2check = (n for n, d in G.nodes(data=True)
                       if n >= 0 and d['subtree'] == subtree)
        if not self.branching:
            nodes2check = (n for n in nodes2check if n == s or len(G[n]) == 1)
        for n in nodes2check:
            # log += F[n]
            for nbr in P.neighbors(n):
                # print(log, F[nbr])
                if nbr < 0:
                    # subtree includes a neighbor of the root
                    print('WARNING: `shortest_gate()` should not be called for'
                          ' nodes whose subtree connects to root through an '
                          "extended Delaunay edge. Something is wrong.")
                    continue
                v = P[nbr][n]['cw']
                if v != P[n][nbr]['ccw']:
                    continue
                u, v = (nbr, v) if nbr < v else (v, nbr)
                # print(n2s(u, v))
                p = self.idx_from_portal.get((u, v))
                if p is not None and p in self.Gmidpt:
                    # log += n2s(*self.portal_from_idx[p])
                    rootsector = self.Gmidpt.nodes[p]['rootsector']
                    subtree_pdict[rootsector].append((p, n))
            # log = ''

        # print(log)
        # print(subtree_pdict)

        # STEP 2) For each rootsector, choose the dominating portal and then
        #         the dominating node. Output is a unique portal sequence per
        #         rootsector.
        min4rootsector = {}
        for rootsector, pn_pairs in subtree_pdict.items():
            # start with the first item as incumbent
            p, n = pn_pairs[0]
            portals = [p]
            incumbent = [n]
            p, = self.Gmidpt.successors(p)
            while p > 0:
                portals.append(p)
                p, = self.Gmidpt.successors(p)
            r = p
            for p, n in pn_pairs[1:]:
                try:
                    i = portals.index(p)
                    if i == 0:
                        # p is tied with the incumbent
                        # TODO: probably this does not happen
                        incumbent.append(n)
                        print('WARNING: incumbent.append(n) actually happens!',
                              *(F[n] for n in incumbent))
                    else:
                        # p dominates the previous incumbent
                        incumbent = [n]
                        portals = portals[i:]
                except ValueError:
                    # p is further away than the incumbent
                    pass
            # now portals[0] is the dominating portal
            # calculate the distances from the portal
            # to each contending node
            dists = np.hypot(*(self.PortalC[portals[0]]
                               - self.VertexC[incumbent]).T)
            # TODO: if the `incumbent.append(n)` indeed does not
            #       happen, incumbent can be a scalar value
            i_min = np.argmin(dists)
            print(f'{i_min},', *[f'{d:.0f}' for d in dists])
            # store: (node, root, sequence of portal indices,
            #         reference distance (through portal midpoints))
            min4rootsector[rootsector] = (incumbent[i_min], r, portals,
                                          self.Gmidpt.nodes[p]['dist'] + dists[i_min])

        # log = ''
        # for k, (n, r, portals, dist) in min4rootsector.items():
            # log += (n2s(*self.portal_from_idx[k])
            #         + f' {F[n]}, {F[r]}, {len(portals)}, {dist:.0f}\n'
        # print(log)

        # STEP 3) Calculate the distances of the shortest paths to root and
        #         pick the path with the shortest distance.
        # final_paths = {}
        contenders = []
        for rootsector, (n, r, portals, pred_dist) in min4rootsector.items():
            portal_pairs = ((self.portal_from_idx[p],
                             self.Gmidpt.nodes[p]['rev'])
                            for p in portals)
            channel = [(u, v) if not rev else (v, u)
                       for (u, v), rev in portal_pairs]
            print('channel:', *(f'{F[s]}–{F[t]}' for s, t in channel))
            path = self.path_by_funnel(n, r, channel)
            print('path:', '–'.join(f'{F[n]}' for n in path))
            segments = (self.VertexC[path[:-1]]
                        - self.VertexC[path[1:]])
            dists = np.hypot(*segments.T)
            actual_dist = dists.sum()
            contenders.append((actual_dist, pred_dist, path,
                               dists, channel[0]))
            # ranking.append((total_dist, pred_dist, rootsector))
            # final_paths[rootsector] = (path, dists, total_dist)

        chosen = min(contenders)

        # STEP 4) Check if the last link crosses any branches of the path tree.
        #         (that can only happen in the first triangle)
        _, _, path, _, (u, v) = chosen
        if len(path) > 2:
            head = path[0]
            tri_side = (u, head) if u < head else (head, u)
            p1 = self.idx_from_portal.get(tri_side)
            if p1 is not None:
                tri_side = (v, head) if v < head else (head, v)
                p2 = self.idx_from_portal.get(tri_side)
                if p2 is not None and (p1 in self.Gmidpt._succ[p2]
                                       or p2 in self.Gmidpt._succ[p1]):
                    print('WARNING: Path crossed in triangle '
                          f'{self.n2s(head, u, v)}')

        return chosen[:-1]
