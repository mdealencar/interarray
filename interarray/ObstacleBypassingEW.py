# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import operator
import time
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .geometric import (angle, angle_helpers, apply_edge_exemptions, assign_root,
                        is_bunch_split_by_corner, is_crossing,
                        is_same_side)
from .crossings import edge_crossings
from .mesh import make_planar_embedding
from .utils import Alerter, NodeStr, NodeTagger
from .priorityqueue import PriorityQueue
from .interarraylib import L_from_G, fun_fingerprint


F = NodeTagger()


def OBEW(L, capacity=8, rootlust=None, maxiter=10000, maxDepth=4,
         MARGIN=1e-4, debug=False, warnwhere=None, weightfun=None):
    '''Obstacle Bypassing Esau-Williams heuristic for C-MST.

    Recommended `rootlust`: '0.6*cur_capacity/capacity'

    Args:
        L: networkx.Graph
        capacity: max number of terminals in a subtree
        rootlust: expression to use for biasing weights
        warnwhere: print debug info based on utils.Alerter

    Returns:
        G_cmst: networkx.Graph
    '''

    start_time = time.perf_counter()

    if warnwhere is not None:
        warn = Alerter(warnwhere, 'i')
    else:  # could check if debug is True and make warn = print
        def warn(*args, **kwargs):
            pass

    # rootlust_formula = '0.7*(cur_capacity/(capacity - 1))**2'
    # rootlust_formula = f'0.6*cur_capacity/capacity'
    # rootlust_formula = f'0.6*(cur_capacity + 1)/capacity'
    # rootlust_formula = f'0.6*cur_capacity/(capacity - 1)'

    if rootlust is None:
        def rootlustfun(_):
            return 0.
    else:
        rootlustfun = eval('lambda cur_capacity: ' + rootlust, locals())
    # rootlust = lambda cur_capacity: 0.7*(cur_capacity/(capacity - 1))**2

    # save relevant options to store in the graph later
    options = dict(MARGIN=MARGIN, variant='C',
                   rootlust=rootlust)

    R, T, B = (L.graph[k] for k in 'RTB')
    roots = range(-R, 0)

    # list of variables indexed by vertex id:
    #     d2roots, d2rootsRank, anglesRank, anglesYhp, anglesXhp
    #     Subtree, VertexC
    # list of variables indexed by subtree id:
    #     CompoIn, CompoLolim, CompoHilim
    # dicts keyed by subtree id
    #     DetourHop, detourLoNotHi
    # sets of subtree ids:
    #     Final_G
    #
    # need to have a table of vertex -> gate node

    # TODO: do away with pre-calculated crossings
    Xings = L.graph.get('crossings')

    # crossings = L.graph['crossings']
    # BEGIN: prepare auxiliary graph with all allowed edges and metrics
    _, A = make_planar_embedding(L)
    assign_root(A)
    P = A.graph['planar']
    diagonals = A.graph['diagonals']
    d2roots = A.graph['d2roots']
    d2rootsRank = rankdata(d2roots, method='dense', axis=0)
    _, anglesRank, anglesXhp, anglesYhp = angle_helpers(A)
    #  triangles = A.graph['triangles']
    #  triangles_exp = A.graph['triangles_exp']
    # apply weightfun on all delaunay edges
    if weightfun is not None:
        # TODO: fix `apply_edge_exemptions()` for the
        #       `delaunay()` without triangles
        apply_edge_exemptions(A)
        options['weightfun'] = weightfun.__name__
        options['weight_attr'] = 'length'
        for _, _, data in A.edges(data=True):
            data['length'] = weightfun(data)
    # removing root nodes from A to speedup find_option4gate
    # this may be done because G already starts with gates
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # BEGIN: create initial star graph
    G = L_from_G(L) if L.number_of_edges() > 0 else L.copy()
    G.add_weighted_edges_from(
        ((n, r, d2roots[n, r]) for n, r in A.nodes(data='root')),
        weight='length')
    nx.set_node_attributes(
        G, {n: r for n, r in A.nodes(data='root')}, 'root')
    # END: create initial star graph

    # BEGIN: helper data structures

    # upper estimate number of Detour nodes:
    # Dmax = round(2*T/3/capacity)
    Dmax = T

    # mappings from nodes
    # <Subtree>: maps nodes to the set of nodes in their subtree
    Subtree = np.empty((T + Dmax,), dtype=object)
    Subtree[:T] = [{n} for n in range(T)]
    # Subtree = np.array([{n} for n in range(T)])
    # TODO: fnT might be better named Pof (Prime of)
    # <fnT>: farm node translation table
    #        to be used when indexing: VertexC, d2roots, angles, etc
    #        fnT[-R..(T+Dmax)] -> -R..T
    fnT = np.arange(T + B + Dmax + R)
    fnT[-R:] = range(-R, 0)

    # <Stale>: list of detour nodes that were discarded
    Stale = []

    # this is to make fnT available for plot animation
    # a new, trimmed, array be assigned after the heuristic is done
    G.graph['fnT'] = fnT

    n2s = NodeStr(fnT, T)
    # <gnT>: gate node translation table
    #        to be used when indexing Subtree[]
    #        gnT[-R..(T+Dmax)] -> -R..T
    gnT = np.arange(T + Dmax)

    # mappings from components (identified by their gates)
    # <ComponIn>: maps component to set of components queued to merge in
    ComponIn = np.array([set() for _ in range(T)], dtype=object)
    ComponLoLim = np.hstack((np.arange(T)[:, np.newaxis],)*R)  # most CW node
    ComponHiLim = ComponLoLim.copy()  # most CW node
    # ComponLoLim = np.arange(T)  # most CCW node
    # ComponHiLim = np.arange(T)  # most CCW node

    # mappings from roots
    # <Final_G>: set of gates of finished components (one set per root)
    Final_G = np.array([set() for _ in range(R)])

    # other structures
    # <pq>: queue prioritized by lowest tradeoff length
    pq = PriorityQueue()
    # find_option4gate()
    # <gates2upd8>: deque for components that need to go through
    # gates2upd8 = deque()
    gates2upd8 = set()
    # <edges2ban>: deque for edges that should not be considered anymore
    # edges2ban = deque()
    # TODO: this is not being used, decide what to do about it
    edges2ban = set()
    VertexC = L.graph['VertexC']
    # number of Detour nodes added
    D = 0
    # <DetourHop>: maps gate nodes to a list of nodes of the Detour path
    #              (root is not on the list)
    DetourHop = defaultdict(list)
    detourLoNotHi = dict()
    # detour = defaultdict(list)
    # detouroverlaps = {}

    # <i>: iteration counter
    i = 0
    # <prevented_crossing>: counter for edges discarded due to crossings
    prevented_crossings = 0
    # END: helper data structures

    def is_rank_within(rank, lowRank, highRank, anglesWrap,
                       touch_is_cross=False):
        less = operator.le if touch_is_cross else operator.lt
        if anglesWrap:
            return less(rank, lowRank) or less(highRank, rank)
        else:
            return less(lowRank, rank) and less(rank, highRank)

    def is_crossing_gate(root, gate, u, v, touch_is_cross=False):
        '''choices for `less`:
        -> operator.lt: touching is not crossing
        -> operator.le: touching is crossing'''
        # get the primes of all nodes
        gate_, u_, v_ = fnT[[gate, u, v]]
        gaterank = anglesRank[gate_, root]
        uR, vR = anglesRank[u_, root], anglesRank[v_, root]
        highRank, lowRank = (uR, vR) if uR >= vR else (vR, uR)
        Xhp = anglesXhp[[u_, v_], root]
        uYhp, vYhp = anglesYhp[[u_, v_], root]
        if is_rank_within(gaterank, lowRank, highRank,
                          not any(Xhp) and uYhp != vYhp, touch_is_cross):
            # TODO: test the funtion's touch_is_cross in the call below
            if not is_same_side(*VertexC[[u_, v_, root, gate_]]):
                # crossing gate
                debug and print(f'<is_crossing_gate> {n2s(u, v)}: would '
                                f'cross gate {n2s(gate)}')
                return True
        return False

    def make_gate_final(root, g2keep):
        if g2keep not in Final_G[root]:
            Final_G[root].add(g2keep)
            log.append((i, 'finalG', (g2keep, root)))
            debug and print(f'<make_gate_final> GATE {n2s(g2keep, root)} added.')

    def component_merging_choices_plain(gate, forbidden=None):
        # gather all the edges leaving the subtree of gate
        if forbidden is None:
            forbidden = set()
        forbidden.add(gate)
        d2root = d2roots[fnT[gate], G.nodes[gate]['root']]
        capacity_left = capacity - len(Subtree[gate])
        weighted_edges = []
        edges2discard = []
        for u in Subtree[gate]:
            for v in A[u]:
                if (gnT[v] in forbidden or
                        len(Subtree[v]) > capacity_left):
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    # newd2root = d2roots[fnT[gnT[v]], G.nodes[fnT[v]]['root']]
                    W = A[u][v]['length']
                    # if W <= d2root:  # TODO: what if I use <= instead of <?
                    if W < d2root:
                        # useful edges
                        #  tiebreaker = d2rootsRank[fnT[v], A[u][v]['root']]
                        tiebreaker = d2rootsRank[fnT[v], A.nodes[v]['root']]
                        weighted_edges.append((W, tiebreaker, u, v))
                        #  weighted_edges.append((W-(d2root - newd2root)/3,
                        #                           tiebreaker, u, v))
        return weighted_edges, edges2discard

    def component_merging_choices(gate, forbidden=None):
        # gather all the edges leaving the subtree of gate
        if forbidden is None:
            forbidden = set()
        forbidden.add(gate)
        root = G.nodes[gate]['root']
        d2root = d2roots[gate, root]
        capacity_left = capacity - len(Subtree[gate])
        root_lust = rootlustfun(len(Subtree[gate]))
        weighted_edges = []
        edges2discard = []
        for u in Subtree[gate]:
            for v in A[u]:
                if (gnT[v] in forbidden or
                        len(Subtree[v]) > capacity_left):
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    d2rGain = d2root - d2roots[gnT[v], G.nodes[fnT[v]]['root']]
                    W = A[u][v]['length']
                    # if W <= d2root:  # TODO: what if I use <= instead of <?
                    if W < d2root:
                        # useful edges
                        #  tiebreaker = d2rootsRank[fnT[v], A[u][v]['root']]
                        tiebreaker = d2rootsRank[fnT[v], A.nodes[v]['root']]
                        # weighted_edges.append((W, tiebreaker, u, v))
                        weighted_edges.append((W - d2rGain*root_lust,
                                               tiebreaker, u, v))
        return weighted_edges, edges2discard

    def sort_union_choices(weighted_edges):
        # this function could be outside esauwilliams()
        unordchoices = np.array(
            weighted_edges,
            dtype=[('weight', np.float64),
                   ('vd2rootR', np.int_),
                   ('u', np.int_),
                   ('v', np.int_)])
        # result = np.argsort(unordchoices, order=['weight'])
        # unordchoices  = unordchoices[result]

        # DEVIATION FROM Esau-Williams
        # rounding of weight to make ties more likely
        # tie-breaking by proximity of 'v' node to root
        # purpose is to favor radial alignment of components
        tempchoices = unordchoices.copy()
        tempchoices['weight'] /= tempchoices['weight'].min()
        tempchoices['weight'] = (20*tempchoices['weight']).round()  # 5%

        result = np.argsort(tempchoices, order=['weight', 'vd2rootR'])
        choices = unordchoices[result]
        return choices

    def find_option4gate(gate):
        debug and i and print(f'<find_option4gate> starting... gate = '
                              f'<{F[gate]}>')
        if edges2ban:
            debug and print(f'<<<<<<<edges2ban>>>>>>>>>>> _{len(edges2ban)}_')
        while edges2ban:
            # edge2ban = edges2ban.popleft()
            edge2ban = edges2ban.pop()
            ban_queued_edge(*edge2ban)
        # () get component expansion edges with weight
        weighted_edges, edges2discard = component_merging_choices(gate)
        # discard useless edges
        A.remove_edges_from(edges2discard)
        # () sort choices
        choices = sort_union_choices(weighted_edges) if weighted_edges else []
        if len(choices) > 0:
            choice = choices[0]
            # merging is better than gate, submit entry to pq
            # weight, u, v = choice
            weight, _, u, v = choice
            # tradeoff calculation
            tradeoff = weight - d2roots[fnT[gate], A.nodes[gate]['root']]
            pq.add(tradeoff, gate, (u, v))
            ComponIn[gnT[v]].add(gate)
            debug and i and print(
                f'<find_option4gate> pushed {n2s(u, v)}, g2drop '
                f'<{F[gate]}>, tradeoff = {-tradeoff:.0f}')
        else:
            # no viable edge is better than gate for this node
            # this becomes a final gate
            if i:  # run only if not at i = 0
                # definitive gates at iteration 0 do not cross any other edges
                # they are not included in Final_G because the algorithm
                # considers the gates extending to infinity (not really)
                root = A.nodes[gate]['root']
                make_gate_final(root, gate)
                # check_heap4crossings(root, gate)
            debug and print('<cancelling>', F[gate])
            if gate in pq.tags:
                # i=0 gates and check_heap4crossings reverse_entry
                # may leave accepting gates out of pq
                pq.cancel(gate)

    def ban_queued_edge(g2drop, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            debug and print('<<<< UNLIKELY <ban_queued_edge()> '
                            f'({F[u]}, {F[v]}) not in A.edges >>>>')
        g2keep = gnT[v]
        # TODO: think about why a discard was needed
        ComponIn[g2keep].discard(g2drop)
        # gates2upd8.appendleft(g2drop)
        gates2upd8.add(g2drop)
        # find_option4gate(g2drop)

        # BEGIN: block to be simplified
        is_reverse = False
        componin = g2keep in ComponIn[g2drop]
        reverse_entry = pq.tags.get(g2keep)
        if reverse_entry is not None:
            _, _, _, (s, t) = reverse_entry
            if (t, s) == (u, v):
                # TODO: think about why a discard was needed
                ComponIn[g2drop].discard(g2keep)
                # this is assymetric on purpose (i.e. not calling
                # pq.cancel(g2drop), because find_option4gate will do)
                pq.cancel(g2keep)
                find_option4gate(g2keep)
                is_reverse = True

        # if this if is not visited, replace the above with ComponIn check
        # this means that if g2keep is to also merge with g2drop, then the
        # edge of the merging must be (v, u)
        if componin != is_reverse:
            # this does happen sometimes (componin: True, is_reverse: False)
            debug and print(f'{n2s(u, v)}, '
                            f'g2drop <{F[g2drop]}>, g2keep <{F[g2keep]}> '
                            f'componin: {componin}, is_reverse: {is_reverse}')

        # END: block to be simplified

    # TODO: check if this function is necessary (not used)
    def abort_edge_addition(g2drop, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            print('<<<< UNLIKELY <abort_edge_addition()> '
                  f'{n2s(u, v)} not in A.edges >>>>')
        ComponIn[gnT[v]].remove(g2drop)
        find_option4gate(g2drop)

    # TODO: check if this function is necessary (not used)
    def get_subtrees_closest_node(gate, origin):
        componodes = list(Subtree[gate])
        if len(componodes) > 1:
            dist = np.squeeze(cdist(VertexC[fnT[componodes]],
                                    VertexC[np.newaxis, fnT[origin]]))
        else:
            dist = np.array([np.hypot(*(VertexC[fnT[componodes[0]]]
                                        - VertexC[np.newaxis, fnT[origin]]).T)])
        idx = np.argmin(dist)
        closest = componodes[idx]
        return closest

    def get_crossings(s, t, detour_waiver=False):
        '''generic crossings checker
        common node is not crossing'''
        s_, t_ = fnT[[s, t]]
        #  st = frozenset((s_, t_))
        st = (s_, t_) if s_ < t_ else (t_, s_)
        #  if st in triangles or st in triangles_exp:
        if st in P.edges or st in diagonals:
            # <(s_, t_) is in the expanded Delaunay edge set>
            #  Xlist = edge_crossings(s_, t_, G, triangles, triangles_exp)
            Xlist = edge_crossings(s_, t_, G, diagonals)
            # crossings with expanded Delaunay already checked
            # just detour edges missing
            nbunch = list(range(T, T + D))
        else:
            # <(s, t) is not in the expanded Delaunay edge set>
            Xlist = []
            nbunch = None  # None means all nodes
        sC, tC = VertexC[[s_, t_]]
        # st_is_detour = s >= T or t >= T
        for w_x in G.edges(nbunch):
            w, x = w_x
            w_, x_ = fnT[[w, x]]
            # both_detours = st_is_detour and (w >= T or x >= T)
            skip = detour_waiver and (w >= T or x >= T)
            if skip or s_ == w_ or t_ == w_ or s_ == x_ or t_ == x_:
                # <edges have a common node>
                continue
            if is_crossing(sC, tC, *VertexC[[w_, x_]], touch_is_cross=True):
                Xlist.append(w_x)
        return Xlist

    def get_crossings_deprecated(s, t):
        # TODO: THIS RELIES ON precalculated crossings
        sC, tC = VertexC[fnT[[s, t]]]
        rootC = VertexC[-R:]
        if np.logical_or.reduce(((sC - rootC)*(tC - rootC)).sum(axis=1) < 0):
            # pre-calculation pruned edges with more than 90° angle
            # so the output will be equivalent to finding a crossings
            return (None,)
        Xlist = []
        for (w, x) in Xings[frozenset(fnT[[s, t]])]:
            if G.has_edge(w, x):
                # debug and print(f'{n2s(w, x)} crosses {n2s(s, t)}')
                Xlist.append((w, x))
        return Xlist

    def plan_detour(root, blocked, goal_, u, v, barrierLo,
                    barrierHi, savings, depth=0, remove=set()):
        '''
        (blocked, goal_) is the detour segment
        (u, v) is an edge crossing it
        barrierLo/Hi are the extremes of the subtree of (u, v) wrt root
        tradeoff = <benefit of the edge addition> - <previous detours>
        '''
        # print(f'[{i}] <plan_detour_recursive[{depth}]> {n2s(u, v)} '
        #       f'blocking {n2s(blocked, goal_)}')
        goalC = VertexC[goal_]
        cleatnode = gnT[blocked]
        detourHop = DetourHop[cleatnode]
        blocked_ = fnT[blocked]
        warn(f'({depth}) ' +
             ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n' if depth == 0
              else '') +
             f'{n2s(u, v)} blocks {n2s(blocked)}, '
             f'gate: {n2s(cleatnode)}')

        # <refL>: length of the edge crossed by (u, v) – reference of cost
        if goal_ < 0:  # goal_ is a root
            refL = d2roots[blocked_, goal_]
        else:
            refL = np.hypot(*(goalC - VertexC[blocked_]).T)
        warn(f'refL: {refL:.0f}')

        is_blocked_a_clone = blocked >= T
        if is_blocked_a_clone:
            blockedHopI = detourHop.index(blocked)
            warn(f'detourHop: {n2s(*detourHop)}')

        # TODO: this would be a good place to handle this special case
        #       but it requires major refactoring of the code
        # if blocked_ in (u, v) and goal_ < 0 and detourHop[-2] >= T:
        if False and blocked_ in (u, v) and goal_ < 0 and detourHop[-2] >= T:
            # <(u, v) edge is actually pushing blocked to one of the limits of
            #  Barrier, this means the actual blocked hop is further away>
            blockedHopI -= 1
            actual_blocked = detourHop[blockedHopI]
            remove = remove | {blocked}
            refL += np.hypot(*(VertexC[fnT[actual_blocked]]
                               - VertexC[blocked_]))
            blocked = actual_blocked
            blocked_ = fnT[blocked]
            is_blocked_a_clone = blocked >= T

        not2hook = remove.copy()

        Barrier = list(Subtree[u] | Subtree[v])

        store = []
        # look for detours on the Lo and Hi sides of barrier
        for corner_, loNotHi, sidelabel in ((barrierLo, True, 'Lo'),
                                            (barrierHi, False, 'Hi')):
            warn(f'({depth}|{sidelabel}) BEGIN: {n2s(corner_)}')

            # block for possible future change (does nothing)
            nearest_root = -R + np.argmin(d2roots[corner_])
            if nearest_root != root:
                debug and print(f'[{i}] corner: {n2s(corner_)} is closest to '
                                f'{n2s(nearest_root)} than to {n2s(root)}')

            # block for finding the best hook
            cornerC = VertexC[corner_]
            Blocked = list(Subtree[cleatnode] - remove)
            if is_blocked_a_clone:
                for j, (hop2check, prevhop) in enumerate(
                        zip(detourHop[blockedHopI::-1],
                            detourHop[blockedHopI - 1::-1])):
                    if j == 2:
                        print(f'[{i}] (2nd iter) depth: {depth}, '
                              f'blocked: {n2s(blocked)}, '
                              f'gate: {n2s(cleatnode)}')
                        # break
                    hop2check_ = fnT[hop2check]
                    is_hop_a_barriers_clone = hop2check_ in Barrier
                    prevhop_ = fnT[prevhop]
                    prevhopC = VertexC[prevhop_]
                    hop2checkC = VertexC[hop2check_]
                    discrim = angle(prevhopC, hop2checkC, cornerC) > 0
                    is_concave_at_hop2check = (
                        (not is_hop_a_barriers_clone and
                         (discrim != detourLoNotHi[hop2check])) or
                        (is_hop_a_barriers_clone and
                         (discrim != loNotHi)))
                    warn(f'concavity check at {n2s(hop2check)}: '
                         f'{is_concave_at_hop2check}')
                    if is_concave_at_hop2check:
                        # warn(f'CONCAVE at {n2s(hop2check)}')
                        #      f'at {n2s(hop2check)}: {is_concave_at_hop2check}, '
                        #      f'remove: {", ".join([F[r] for r in remove])}')
                        if hop2check not in remove:
                            if prevhop >= T and hop2check not in Barrier:
                                prevprevhop = detourHop[blockedHopI - j - 2]
                                prevprevhopC = VertexC[fnT[prevprevhop]]
                                prevhopSubTreeC = \
                                    VertexC[[h for h in Subtree[prevhop_]
                                             if h < T]]
                                # TODO: the best thing would be to use here the
                                #       same split algorithm used later
                                #       (barrsplit)
                                if is_bunch_split_by_corner(
                                        prevhopSubTreeC, cornerC,
                                        prevhopC, prevprevhopC)[0]:
                                    break
                            # get_crossings(corner_, prevhop_)):
                            # <detour can actually bypass the previous one>
                            Blocked.remove(hop2check)
                            not2hook |= {hop2check}
                    else:
                        break
            # get the distance from every node in Blocked to corner
            D2corner = np.squeeze(cdist(VertexC[fnT[Blocked]],
                                        VertexC[np.newaxis, corner_]))
            if not D2corner.shape:
                D2corner = [float(D2corner)]
            hookI = np.argmin(D2corner)
            hook = Blocked[hookI]
            hookC = VertexC[fnT[hook]]

            # block for calculating the length of the path to replace
            prevL = refL
            shift = hook != blocked
            if shift and is_blocked_a_clone:
                prevhop_ = blocked_
                for hop in detourHop[blockedHopI - 1::-1]:
                    hop_ = fnT[hop]
                    prevL += np.hypot(*(VertexC[hop_] -
                                        VertexC[prevhop_]))
                    warn(f'adding {n2s(hop_, prevhop_)}')
                    prevhop_ = hop_
                    if hop == hook or hop < T:
                        break
            warn(f'prevL: {prevL:.0f}')

            # check if the bend at corner is necessary
            discrim = angle(hookC, cornerC, goalC) > 0
            dropcorner = discrim != loNotHi
            # if hook < T and dropcorner:
            # if dropcorner and False:  # TODO: conclude this test
            if dropcorner and fnT[hook] != corner_:
                warn(f'DROPCORNER {sidelabel}')
                # <bend unnecessary>
                detourL = np.hypot(*(goalC - hookC))
                addedL = prevL - detourL
                # debug and print(f'[{i}] CONCAVE: '
                # print(f'[{i}] CONCAVE: '
                #       f'{n2s(hook, corner_, goal_)}')
                detourX = get_crossings(goal_, hook, detour_waiver=True)
                if not detourX:
                    path = (hook,)
                    LoNotHi = tuple()
                    direct = True
                    store.append((addedL, path, LoNotHi, direct, shift))
                    continue
            warn(f'hook: {n2s(hook)}')
            nearL = (d2roots[corner_, goal_] if goal_ < 0
                     else np.hypot(*(goalC - cornerC)))
            farL = D2corner[hookI]
            addedL = farL + nearL - prevL
            warn(f'{n2s(hook, corner_, goal_)} '
                 f'addedL: {addedL:.0f}')
            if addedL > savings:
                # <detour is more costly than the savings from (u, v)>
                store.append((np.inf, (hook, corner_)))
                continue

            # TODO: the line below is risky. it disconsiders detour nodes
            #       when checking if a subtree is split
            BarrierPrime = np.array([b for b in Barrier if b < T])
            BarrierC = VertexC[fnT[BarrierPrime]]

            is_barrier_split, insideI, outsideI = is_bunch_split_by_corner(
                BarrierC, hookC, cornerC, goalC)

            # TODO: think if this gate edge waiver is correct
            FarX = [farX for farX in get_crossings(hook, corner_,
                                                   detour_waiver=True)
                    if farX[0] >= 0 and farX[1] >= 0]
            # this will condense multiple edges from the same subtree into one
            FarXsubtree = {gnT[s]: (s, t) for s, t in FarX}

            # BEGIN barrHack block
            Nin, Nout = len(insideI), len(outsideI)
            if is_barrier_split and (Nin == 1 or Nout == 1):
                # <possibility of doing the barrHack>
                barrHackI = outsideI[0] if Nout <= Nin else insideI[0]
                barrierX_ = BarrierPrime[barrHackI]
                # TODO: these criteria are too ad hoc, improve it
                if gnT[barrierX_] in FarXsubtree:
                    del FarXsubtree[gnT[barrierX_]]
                elif (barrierX_ not in G[corner_]
                      and d2roots[barrierX_, root] >
                        1.1*d2roots[fnT[hook], root]):
                    # <this is a spurious barrier split>
                    # ignore this barrier split
                    is_barrier_split = False
                    warn('spurious barrier split detected')
            else:
                barrHackI = None
            # END barrHack block

            # possible check to include: (something like this)
            # if (anglesRank[corner_, root] >
            #     anglesRank[ComponHiLim[gnT[blocked], root]]
            warn(f'barrsplit: {is_barrier_split}, inside: {len(insideI)}, '
                 f'outside: {len(outsideI)}, total: {len(BarrierC)}')
            # if is_barrier_split or get_crossings(corner_, goal_):
            barrAddedL = 0
            nearX = get_crossings(corner_, goal_, detour_waiver=True)
            if nearX:
                warn(f'nearX: {", ".join(n2s(*X) for X in nearX)}')
            if nearX or (is_barrier_split and barrHackI is None):
                # <barrier very split or closer segment crosses some edge>
                store.append((np.inf, (hook, corner_)))
                continue
            # elif (is_barrier_split and len(outsideI) == 1 and
            #       len(insideI) > 3):
            elif is_barrier_split:
                # <barrier slightly split>
                # go around small interferences with the barrier itself
                warn(f'SPLIT: {n2s(hook, corner_)} leaves '
                     f'{n2s(BarrierPrime[barrHackI])} isolated')
                # will the FarX code handle this case?
                barrierXC = BarrierC[barrHackI]
                # barrpath = (hook, barrierX, corner_)
                # two cases: barrHop before or after corner_
                corner1st = (d2rootsRank[barrierX_, root] <
                             d2rootsRank[corner_, root])
                if corner1st:
                    barrAddedL = (np.hypot(*(goalC - barrierXC))
                                  + np.hypot(*(cornerC - barrierXC))
                                  - nearL)
                    barrhop = (barrierX_,)
                else:
                    barrAddedL = (np.hypot(*(hookC - barrierXC))
                                  + np.hypot(*(cornerC - barrierXC))
                                  - farL)
                    barrhop = (corner_,)
                    corner_ = barrierX_
                warn(f'barrAddedL: {barrAddedL:.0f}')
                addedL += barrAddedL
                barrLoNotHi = (loNotHi,)
            else:
                barrhop = tuple()
                barrLoNotHi = tuple()

            if len(FarXsubtree) > 1:
                warn(f'NOT IMPLEMENTED: many ({len(FarXsubtree)}) '
                     f'crossings of {n2s(hook, corner_)} ('
                     f'{", ".join([n2s(a, b) for a, b in FarXsubtree.values()])})')
                store.append((np.inf, (hook, corner_)))
                continue
            elif FarXsubtree:  # there is one crossing
                subbarrier, farX = FarXsubtree.popitem()
                # print('farX:', n2s(*farX))
                if depth > maxDepth:
                    print(f'<plan_detour[{depth}]> max depth ({maxDepth})'
                          'exceeded.')
                    store.append((np.inf, (hook, corner_)))
                    continue
                else:
                    new_barrierLo = ComponLoLim[subbarrier, root]
                    new_barrierHi = ComponHiLim[subbarrier, root]
                    remaining_savings = savings - addedL
                    subdetour = plan_detour(
                        root, hook, corner_, *farX, new_barrierLo,
                        new_barrierHi, remaining_savings, depth + 1,
                        remove=not2hook)
                    if subdetour is None:
                        store.append((np.inf, (hook, corner_)))
                        continue
                    subpath, subaddedL, subLoNotHi, subshift = subdetour

                    subcorner = subpath[-1]
                    # TODO: investigate why plan_detour is suggesting
                    #       hops that are not primes
                    subcorner_ = fnT[subcorner]
                    subcornerC = VertexC[subcorner_]
                    # check if the bend at corner is necessary
                    nexthopC = fnT[barrhop[0]] if barrhop else goalC
                    discrim = angle(subcornerC, cornerC, nexthopC) > 0
                    dropcorner = discrim != loNotHi
                    # TODO: come back to this False
                    if False and dropcorner:
                        subcornerC = VertexC[subcorner_]
                        dropL = np.hypot(*(nexthopC - subcornerC))
                        dc_addedL = dropL - prevL + barrAddedL
                        direct = len(subpath) == 1
                        if not direct:
                            subfarL = np.hypot(*(cornerC - subcornerC))
                            subnearL = subaddedL - subfarL + nearL
                            dc_addedL += subnearL
                        # debug and print(f'[{i}] CONCAVE: '
                        # print(f'[{i}] CONCAVE: '
                        #       f'{n2s(hook, corner_, goal_)}')
                        dcX = get_crossings(subcorner_, corner_,
                                            detour_waiver=True)
                        if not dcX:
                            print(f'[{i}, {depth}] dropped corner '
                                  f'{n2s(corner_)}')
                            path = (*subpath, *barrhop)
                            LoNotHi = (*subLoNotHi, *barrLoNotHi)
                            store.append((dc_addedL, path, LoNotHi,
                                          direct, shift))
                            continue

                    # combine the nested detours
                    path = (*subpath, corner_, *barrhop)
                    LoNotHi = (*subLoNotHi, *barrLoNotHi)
                    addedL += subaddedL
                    shift = subshift
            else:  # there are no crossings
                path = (hook, corner_, *barrhop)
                LoNotHi = (*barrLoNotHi,)
            warn(f'{sidelabel} STORE: {n2s(*path)}'
                 f' addedL: {addedL:.0f}')
            # TODO: properly check for direct connection
            # TODO: if shift: test if there is a direct path
            #       from hook to root
            direct = False
            # TODO: deprecate shift?
            store.append((addedL, path, LoNotHi, direct, shift))

        # choose between the low or high corners
        if store[0][0] < savings or store[1][0] < savings:
            loNotHi = store[0][0] < store[1][0]
            cost, path, LoNotHi, direct, shift = store[not loNotHi]
            warn(f'({depth}) '
                 f'take: {n2s(*store[not loNotHi][1], goal_)} (@{cost:.0f}), '
                 f'drop: {n2s(*store[loNotHi][1], goal_)} '
                 f'(@{store[loNotHi][0]:.0f})')
            debug and print(f'<plan_detour[{depth}]>: {n2s(u, v)} crosses '
                            f'{n2s(blocked, goal_)} but {n2s(*path, goal_)} '
                            'may be used instead.')
            return (path, cost, LoNotHi + (loNotHi,), shift)
        return None

    def add_corner(hook, corner_, cleatnode, loNotHi):
        nonlocal D
        D += 1

        if D > Dmax:
            # TODO: extend VertexC, fnT and gnT
            print('@@@@@@@@@@@@@@ Dmax REACHED @@@@@@@@@@@@@@')
        corner = T + B + D - 1

        # update coordinates mapping fnT
        fnT[corner] = corner_

        # update gates mapping gnT
        # subtree being rerouted
        gnT[corner] = cleatnode
        Subtree[cleatnode].add(corner)
        Subtree[corner] = Subtree[cleatnode]

        # update DetourHop
        DetourHop[cleatnode].append(corner)
        # update detourLoNotHi
        detourLoNotHi[corner] = loNotHi
        # add Detour node
        G.add_node(corner, kind='detour', root=G.nodes[hook]['root'])
        log.append((i, 'addDN', (corner_, corner)))
        # add detour edges
        length = np.hypot(*(VertexC[fnT[hook]] -
                            VertexC[corner_]).T)
        G.add_edge(hook, corner, length=length,
                   kind='detour', color='yellow', style='dashed')
        log.append((i, 'addDE', (hook, corner, fnT[hook], corner_)))
        return corner

    def move_corner(corner, hook, corner_, cleatnode, loNotHi):
        # update translation tables
        fnT[corner] = corner_

        # update DetourHop
        DetourHop[cleatnode].append(corner)
        # update detourLoNotHi
        detourLoNotHi[corner] = loNotHi
        # update edges lengths
        farL = np.hypot(*(VertexC[fnT[hook]]
                          - VertexC[corner_]).T)
        # print(f'[{i}] updating {n2s(hook, corner)}')
        G[hook][corner].update(length=farL)
        log.append((i, 'movDN', (hook, corner, fnT[hook], corner_)))
        return corner

    def make_detour(blocked, path, LoNotHi, shift):
        hook, *Corner_ = path

        cleatnode = gnT[blocked]
        root = G.nodes[cleatnode]['root']
        # if Corner_[0] is None:
        if not Corner_:
            # <a direct gate replacing previous gate>
            # TODO: this case is very outdated, probably wrong
            debug and print(f'[{i}] <make_detour> direct gate '
                            f'{n2s(hook, root)}')
            # remove previous gate
            Final_G[root].remove(blocked)
            Subtree[cleatnode].remove(blocked)
            G.remove_edge(blocked, root)
            log.append((i, 'remE', (blocked, root)))
            # make a new direct gate
            length = d2roots[fnT[hook], root]
            G.add_edge(hook, root, length=length,
                       kind='detour', color='yellow', style='dashed')
            log.append((i, 'addDE', (hook, root, fnT[hook], root)))
            Final_G[root].add(hook)
        else:
            detourHop = DetourHop[cleatnode]
            if blocked < T or hook == blocked:
                # <detour only affects the blocked gate edge>

                # remove the blocked gate edge
                Final_G[root].remove(blocked)
                G.remove_edge(blocked, root)
                log.append((i, 'remE', (blocked, root)))

                # create new corner nodes
                if hook < T:
                    # add the first entry in DetourHop (always prime)
                    detourHop.append(hook)
                for corner_, loNotHi in zip(Corner_, LoNotHi):
                    corner = add_corner(hook, corner_, cleatnode, loNotHi)
                    hook = corner
                # add the gate edge from the last corner node created
                length = d2roots[corner_, root]
                G.add_edge(corner, root, length=length,
                           kind='detour', color='yellow', style='dashed')
                log.append((i, 'addDE', (corner, root, corner_, root)))
                Final_G[root].add(corner)
            else:
                # <detour affects edges further from blocked node>

                assert blocked == detourHop[-1]
                # stales = iter(detourHop[-2:0:-1])

                # number of new corners needed
                newN = len(Corner_) - len(detourHop) + 1

                try:
                    j = detourHop.index(hook)
                except ValueError:
                    # <the path is starting from a new prime>
                    j = 0
                    stales = iter(detourHop[1:])
                    k = abs(newN) if newN < 0 else 0
                    new2stale_cut = detourHop[k:k + 2]
                    detourHop.clear()
                    detourHop.append(hook)
                else:
                    stales = iter(detourHop[j + 1:])
                    newN += j
                    k = j + (abs(newN) if newN < 0 else 0)
                    new2stale_cut = detourHop[k:k + 2]
                    del detourHop[j + 1:]
                # print(f'[{i}] <make_detour> removing {n2s(*new2stale_cut)}, '
                #       f'{new2stale_cut in G.edges}')
                # newN += j
                # TODO: this is not handling the case of more stale hops than
                #       necessary for the detour path (must at least cleanup G)
                if newN < 0:
                    print(f'[{i}] WARNING <make_detour> more stales than '
                          f'needed: {abs(newN)}')
                    while newN < 0:
                        stale = next(stales)
                        G.remove_node(stale)
                        log.append((i, 'remN', stale))
                        Subtree[cleatnode].remove(stale)
                        Stale.append(stale)
                        newN += 1
                else:
                    G.remove_edge(*new2stale_cut)
                    log.append((i, 'remE', new2stale_cut))
                for j, (corner_, loNotHi) in enumerate(zip(Corner_, LoNotHi)):
                    if j < newN:
                        # create new corner nodes
                        corner = add_corner(hook, corner_, cleatnode, loNotHi)
                    else:
                        stale = next(stales)
                        if j == newN:
                            # add new2stale_cut edge
                            # print(f'[{i}] adding {n2s(hook, stale)}')
                            G.add_edge(hook, stale, kind='detour',
                                       color='yellow', style='dashed')
                            log.append((i, 'addDE', (hook, stale,
                                                     fnT[hook], corner_)))
                        # move the stale corners to their new places
                        corner = move_corner(stale, hook, corner_,
                                             cleatnode, loNotHi)
                    hook = corner
                # update the gate edge length
                nearL = d2roots[corner_, root]
                G[corner][root].update(length=nearL)
                log.append((i, 'movDN', (corner, root, corner_, root)))

    def check_gate_crossings(u, v, g2keep, g2drop):
        nonlocal tradeoff

        union = list(Subtree[u] | Subtree[v])
        r2keep = G.nodes[g2keep]['root']
        r2drop = G.nodes[g2drop]['root']

        if r2keep == r2drop:
            roots2check = (r2keep,)
        else:
            roots2check = (r2keep, r2drop)

        # assess the union's angle span
        unionHi = np.empty((len(roots),), dtype=int)
        unionLo = np.empty((len(roots),), dtype=int)
        for root in roots:
            keepHi = ComponHiLim[g2keep, root]
            keepLo = ComponLoLim[g2keep, root]
            dropHi = ComponHiLim[g2drop, root]
            dropLo = ComponLoLim[g2drop, root]
            unionHi[root] = (
                dropHi if angle(*VertexC[fnT[[keepHi, root, dropHi]]]) > 0
                else keepHi)
            unionLo[root] = (
                dropLo if angle(*VertexC[fnT[[dropLo, root, keepLo]]]) > 0
                else keepLo)
            # debug and print(f'<angle_span> //{F[unionLo]} : '
            #                 f'{F[unionHi]}//')

        abort = False
        Detour = {}

        for root in roots2check:
            for g2check in Final_G[root] - {v}:
                if (is_crossing_gate(root, g2check, u, v,
                                     touch_is_cross=True) or
                        (g2check >= T and fnT[g2check] in (u, v) and
                         (is_bunch_split_by_corner(
                             VertexC[fnT[union]],
                             *VertexC[fnT[[DetourHop[gnT[g2check]][-2],
                                          g2check, root]]])[0]))):
                    # print('getting detour')
                    # detour = plan_detour(root, g2check,
                    #                     u, v, unionLo[root],
                    #                     unionHi[root], -tradeoff)
                    # TODO: it would be worth checking if changing roots is the
                    #       shortest path to avoid the (u, v) block
                    detour = plan_detour(
                        root, g2check, root, u, v,
                        unionLo[root], unionHi[root], -tradeoff)
                    if detour is not None:
                        Detour[g2check] = detour
                    else:
                        debug and print(f'<check_gate_crossings> discarding '
                                        f'{n2s(u, v)}: '
                                        f'would block gate {n2s(g2check)}')
                        abort = True
                        break
            if abort:
                break

        if not abort and Detour:
            debug and print(
                f'<check_gate_crossings> detour options: '
                f'{", ".join(n2s(*path) for path, _, _, _ in Detour.values())}')
            # <crossing detected but detours are possible>
            detoursCost = sum((cost for _, cost, _, _ in Detour.values()))
            if detoursCost < -tradeoff:
                # add detours to G
                detdesc = [f'blocked {n2s(blocked)}, '
                           f'gate {n2s(gnT[blocked])}, '
                           f'{n2s(*path)} '
                           f'@{cost:.0f}'
                           for blocked, (path, cost, loNotHi, shift)
                           in Detour.items()]
                warn('\n' + '\n'.join(detdesc))
                for blocked, (path, _, LoNotHi, shift) in Detour.items():
                    make_detour(blocked, path, LoNotHi, shift)
            else:
                debug and print(
                    f'Multiple Detour cancelled for {n2s(u, v)} '
                    f'(tradeoff gain = {-tradeoff:.0f}) × '
                    f'(cost = {detoursCost:.0f}):\n'  # +
                    # '\n'.join(detourdesc))
                    )
                abort = True
        return abort, unionLo, unionHi

    # initialize pq
    for n in range(T):
        find_option4gate(n)
    # create a global tradeoff variable
    tradeoff = 0

    # BEGIN: main loop
    def loop():
        '''Takes a step in the iterative tree building process.
        Return value [bool]: not done.'''
        nonlocal i, prevented_crossings, tradeoff
        while True:
            i += 1
            if i > maxiter:
                print(f'ERROR: maxiter reached ({i})')
                return
            if gates2upd8:
                debug and print('<loop> gates2upd8:',
                                ', '.join(F[gate] for gate in gates2upd8))
            while gates2upd8:
                find_option4gate(gates2upd8.pop())
            if not pq:
                # finished
                return
            tradeoff = pq[0][0]
            debug and print(f'\n[{i}] tradeoff gain = {-tradeoff:.0f}')
            g2drop, (u, v) = pq.top()
            debug and print(f'<loop> POPPED {n2s(u, v)},'
                            f' g2drop: <{F[g2drop]}>')
            capacity_left = capacity - len(Subtree[u]) - len(Subtree[v])

            if capacity_left < 0:
                print('@@@@@ Does this ever happen?? @@@@@')
                ban_queued_edge(g2drop, u, v)
                yield (u, v), False
                continue

            # BEGIN edge crossing check
            # check if (u, v) crosses an existing edge
            # look for crossing edges within the neighborhood of (u, v)
            # only works if using the expanded delaunay edges
            #  eX = edge_crossings(u, v, G, triangles, triangles_exp)
            eX = edge_crossings(u, v, G, diagonals)
            # Detour edges need to be checked separately
            if not eX and D:
                uC, vC = VertexC[fnT[[u, v]]]
                eXtmp = []
                eXnodes = set()
                nodes2check = set()
                BarrierC = VertexC[fnT[list(Subtree[u] | Subtree[v])]]
                for s, t in G.edges(range(T, T + D)):
                    skip = False
                    if s < 0 or t < 0:
                        # skip gates (will be checked later)
                        continue
                    s_, t_ = fnT[[s, t]]
                    Corner = []
                    # below are the 2 cases in which a new edge
                    # will join two subtrees across a detour edge
                    if (s >= T and (s_ == u or s_ == v) and
                            s != DetourHop[gnT[s]][-1]):
                        Corner.append(s)
                    if (t >= T and (t_ == u or t_ == v) and
                            t != DetourHop[gnT[t]][-1]):
                        Corner.append(t)
                    for corner in Corner:
                        a, b = G[corner]
                        if is_bunch_split_by_corner(
                                BarrierC,
                                *VertexC[fnT[[a, corner, b]]])[0]:
                            debug and print(f'[{i}] {n2s(u, v)} '
                                            'would cross '
                                            f'{n2s(a, corner, b)}')
                            eX.append((a, corner, b))
                            skip = True
                    if skip:
                        continue
                    if is_crossing(uC, vC, *VertexC[fnT[[s, t]]],
                                   touch_is_cross=False):
                        eXtmp.append((s, t))
                        if s in eXnodes:
                            nodes2check.add(s)
                        if t in eXnodes:
                            nodes2check.add(t)
                        eXnodes.add(s)
                        eXnodes.add(t)
                for s, t in eXtmp:
                    if s in nodes2check:
                        for w in G[s]:
                            if w != t and not is_same_side(
                                    uC, vC, *VertexC[fnT[[w, t]]]):
                                eX.append((s, t))
                    elif t in nodes2check:
                        for w in G[t]:
                            if w != s and not is_same_side(
                                    uC, vC, *VertexC[fnT[[w, s]]]):
                                eX.append((s, t))
                    else:
                        eX.append((s, t))

            if eX:
                debug and print(f'<edge_crossing> discarding {n2s(u, v)} – would '
                                f'cross: {", ".join(n2s(s, t) for s, t in eX)}')
                # abort_edge_addition(g2drop, u, v)
                prevented_crossings += 1
                ban_queued_edge(g2drop, u, v)
                yield (u, v), None
                continue
            # END edge crossing check

            # BEGIN gate crossing check
            # check if (u, v) crosses an existing gate
            g2keep = gnT[v]
            root = G.nodes[g2keep]['root']
            r2drop = G.nodes[g2drop]['root']
            if root != r2drop:
                debug and print(f'<distinct_roots>: [{F[u]}] is bound to '
                                f'[{F[r2drop]}], while'
                                f'[{F[v]}] is to {F[root]}.')

            abort, unionLo, unionHi = check_gate_crossings(u, v, g2keep,
                                                           g2drop)
            if abort:
                prevented_crossings += 1
                ban_queued_edge(g2drop, u, v)
                yield (u, v), None
                continue
            # END gate crossing check

            # (u, v) edge addition starts here
            subtree = Subtree[v]
            subtree |= Subtree[u]
            G.remove_edge(A.nodes[u]['root'], g2drop)
            log.append((i, 'remE', (A.nodes[u]['root'], g2drop)))

            g2keep_entry = pq.tags.get(g2keep)
            if g2keep_entry is not None:
                _, _, _, (_, t) = g2keep_entry
                # print('node', F[t], 'gate', F[gnT[t]])
                ComponIn[gnT[t]].remove(g2keep)
            # TODO: think about why a discard was needed
            ComponIn[g2keep].discard(g2drop)

            # update the component's angle span
            ComponHiLim[g2keep] = unionHi
            ComponLoLim[g2keep] = unionLo

            # assign root, gate and subtree to the newly added nodes
            for n in Subtree[u]:
                A.nodes[n]['root'] = root
                G.nodes[n]['root'] = root
                gnT[n] = g2keep
                Subtree[n] = subtree
            debug and print(f'<loop> NEW EDGE {n2s(u, v)}, g2keep '
                            f'<{F[g2keep]}>, ' if pq else 'EMPTY heap')
            #  G.add_edge(u, v, **A.edges[u, v])
            G.add_edge(u, v, length=A[u][v]['length'])
            log.append((i, 'addE', (u, v)))
            # remove from consideration edges internal to Subtree
            A.remove_edge(u, v)

            # finished adding the edge, now check the consequences
            if capacity_left > 0:
                for gate in list(ComponIn[g2keep]):
                    if len(Subtree[gate]) > capacity_left:
                        # <this subtree got too big for Subtree[gate] to join>
                        ComponIn[g2keep].discard(gate)
                        gates2upd8.add(gate)
                # for gate in ComponIn[g2drop]:
                for gate in ComponIn[g2drop] - ComponIn[g2keep]:
                    if len(Subtree[gate]) > capacity_left:
                        gates2upd8.add(gate)
                    else:
                        ComponIn[g2keep].add(gate)
                gates2upd8.add(g2keep)
            else:
                # max capacity reached: subtree full
                if g2keep in pq.tags:  # required because of i=0 gates
                    pq.cancel(g2keep)
                make_gate_final(root, g2keep)
                # don't consider connecting to this full subtree anymore
                A.remove_nodes_from(subtree)
                # for gate in ((ComponIn[g2drop] | ComponIn[g2keep]) - {g2drop,
                # g2keep}):
                for gate in ComponIn[g2drop] | ComponIn[g2keep]:
                    gates2upd8.add(gate)
    # END: main loop

    log = []
    G.graph['log'] = log
    for _ in loop():
        pass

    if Stale:
        debug and print(f'Stale nodes ({len(Stale)}):', [n2s(n) for n in Stale])
        old2new = np.arange(T + B, T + B + D)
        mask = np.ones(D, dtype=bool)
        for s in Stale:
            old2new[s - T - B + 1:] -= 1
            mask[s - T - B] = False
        mapping = dict(zip(range(T + B, T + B + D), old2new))
        for k in Stale:
            mapping.pop(k)
        nx.relabel_nodes(G, mapping, copy=False)
        fnT[T + B:T + B + D - len(Stale)] = fnT[T + B:T + B + D][mask]
        D -= len(Stale)

    debug and print(f'FINISHED – Detour nodes added: {D}')

    if debug:
        not_marked = []
        for root in roots:
            for gate in G[root]:
                if gate not in Final_G[root]:
                    not_marked.append(gate)
        not_marked and print('@@@@ WARNING: gates '
                             f'<{", ".join([F[gate] for gate in not_marked])}'
                             '> were not marked as final @@@@')

    # algorithm finished, store some info in the graph object
    G.graph.update(
        creator='OBEW',
        capacity=capacity,
        runtime=time.perf_counter() - start_time,
        d2roots=d2roots,
        method_options= options | dict(
            fun_fingerprint=fun_fingerprint(),
        ),
        solver_details=dict(
            iterations=i,
            prevented_crossings=prevented_crossings,
        ),
    )
    if D > 0:
        G.graph['D'] = D
        G.graph['fnT'] = np.concatenate((fnT[:T + B + D], fnT[-R:]))

    return G
