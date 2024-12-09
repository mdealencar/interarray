# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import operator
import time

import numpy as np
import networkx as nx
from scipy.stats import rankdata

from .mesh import delaunay
from .geometric import (angle, apply_edge_exemptions, assign_root, complete_graph,
                        is_crossing, is_same_side, angle_helpers)
from .crossings import edge_crossings
from .utils import NodeTagger
from .priorityqueue import PriorityQueue


F = NodeTagger()


def CPEW(G_base, capacity=8, delaunay_based=True, maxiter=10000,
         debug=False, weightfun=None, weight_attr='length'):
    '''Crossing Preventing Esau-Williams heuristic for C-MST
    inputs:
    G_base: networkx.Graph
    c: capacity
    returns G_cmst: networkx.Graph'''

    start_time = time.perf_counter()
    # grab relevant options to store in the graph later
    options = dict(delaunay_based=delaunay_based)

    R = G_base.graph['R']
    T = G_base.graph['T']
    roots = range(-R, 0)
    VertexC = G_base.graph['VertexC']

    # BEGIN: prepare auxiliary graph with all allowed edges and metrics
    if delaunay_based:
        A = delaunay(G_base, bind2root=True)
        P = A.graph['planar']
        diagonals = A.graph['diagonals']
        #  A = delaunay_deprecated(G_base)
        #  triangles = A.graph['triangles']
        #  triangles_exp = A.graph['triangles_exp']
        # apply weightfun on all delaunay edges
        if weightfun is not None:
            # TODO: fix `apply_edge_exemptions()` for the
            #       `delaunay()` without triangles
            apply_edge_exemptions(A)
        # TODO: decide whether to keep this 'else' (to get edge arcs)
        # else:
            # apply_edge_exemptions(A)
    else:
        A = complete_graph(G_base)

    assign_root(A)
    d2roots = A.graph.get('d2roots')
    d2rootsRank = rankdata(d2roots, method='dense', axis=0)
    _, anglesRank, anglesXhp, anglesYhp = angle_helpers(G_base)

    if weightfun is not None:
        options['weightfun'] = weightfun.__name__
        options['weight_attr'] = weight_attr
        for u, v, data in A.edges(data=True):
            data[weight_attr] = weightfun(data)
    # removing root nodes from A to speedup find_option4gate
    # this may be done because G already starts with gates
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # BEGIN: create initial star graph
    G = nx.create_empty_copy(G_base)
    G.add_weighted_edges_from(
        ((n, r, d2roots[n, r]) for n, r in A.nodes(data='root') if n >= 0),
        weight=weight_attr)
    # END: create initial star graph

    # BEGIN: helper data structures

    # mappings from nodes
    # <subtrees>: maps nodes to the set of nodes in their subtree
    subtrees = np.array([{n} for n in range(T)])
    # <Gate>: maps nodes to their gates
    Gate = np.array([n for n in range(T)])

    # mappings from components (identified by their gates)
    # <ComponIn>: maps component to set of components queued to merge in
    ComponIn = np.array([set() for _ in range(T)])
    ComponLoLim = np.arange(T)  # most CW node
    ComponHiLim = np.arange(T)  # most CCW node

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
    # TODO: edges{T,C,V} could be used to vectorize the edge crossing detection
    # <edgesN>: array of nodes of the edges of G (T×2)
    # <edgesC>: array of node coordinates for edges of G (T×2×2)
    # <edgesV>: array of vectors of the edges of G (T×2)
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
        gaterank = anglesRank[gate, root]
        uR, vR = anglesRank[u, root], anglesRank[v, root]
        highRank, lowRank = (uR, vR) if uR >= vR else (vR, uR)
        Xhp = anglesXhp[[u, v], root]
        uYhp, vYhp = anglesYhp[[u, v], root]
        if is_rank_within(gaterank, lowRank, highRank,
                          not any(Xhp) and uYhp != vYhp, touch_is_cross):
            if not is_same_side(*VertexC[[u, v, root, gate]]):
                # crossing gate
                debug and print(f'<crossing> discarding '
                                f'«{F[u]}–{F[v]}»: would cross'
                                f'gate <{F[gate]}>')
                return True
        return False

    def make_gate_final(root, g2keep):
        Final_G[root].add(g2keep)
        log.append((i, 'finalG', (g2keep, root)))
        debug and print(f'<final> gate '
                        f'[{F[g2keep]}] added')

    def component_merging_choices(gate, forbidden=None):
        # gather all the edges leaving the subtree of gate
        if forbidden is None:
            forbidden = set()
        forbidden.add(gate)
        d2root = d2roots[gate, A.nodes[gate]['root']]
        capacity_left = capacity - len(subtrees[gate])
        weighted_edges = []
        edges2discard = []
        for u in subtrees[gate]:
            for v in A[u]:
                if (Gate[v] in forbidden or
                        len(subtrees[v]) > capacity_left):
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    W = A[u][v][weight_attr]
                    # if W <= d2root:  # TODO: what if I use <= instead of <?
                    if W < d2root:
                        # useful edges
                        tiebreaker = d2rootsRank[v, A[u][v]['root']]
                        weighted_edges.append((W, tiebreaker, u, v))
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

    def first_non_crossing(choices, gate):
        '''go through choices and return the first that does not cross a final
        gate'''
        # TODO: remove gate from the parameters
        nonlocal prevented_crossings
        found = False
        # BEGIN: for loop that picks an edge
        for weight, tiebreaker, u, v in choices:
            found = True
            root = A[u][v]['root']

            # check if a gate is crossing the edge (u, v)
            # TODO: this only looks at the gates connecting to the edges'
            # closest root , is it relevant to look at all roots?
            # PendingG = set()

            for finalG in Final_G[root]:
                # TODO: test a gate exactly overlapping with a node
                # Elaborating: angleRank will take care of this corner case.
                # the gate will fall within one of the edges around the node
                if is_crossing_gate(root, finalG, u, v):
                    # crossing gate, discard edge
                    prevented_crossings += 1
                    # TODO: call ban_queued_edge (problem: these edges are not
                    # queued)
                    if (u, v) in A.edges:
                        A.remove_edge(u, v)
                    else:
                        debug and print('<<<<<< UNLIKELY.A first_non_crossing'
                                        f'(): ({F[u]}, {F[v]}) not in '
                                        'A.edges >>>>')
                    if Gate[v] in ComponIn[gate]:
                        # this means the target component was in line to
                        # connect to the current component
                        debug and print('<<<<<< UNLIKELY.B first_non_crossing'
                                        f'(): Gate[{F[v]}] in ComponIn['
                                        f'{F[gate]}] >>>>>>>')
                        _, _, _, (s, t) = pq.tags.get(Gate[v])
                        if t == u:
                            ComponIn[gate].remove(Gate[v])
                            # PendingG.add(Gate[v])
                            # gates2upd8.append(Gate[v])
                            gates2upd8.add(Gate[v])

                    found = False
                    break
            # for pending in PendingG:
                #  print(f'<pending> processing '
                #        f'pending [{F[pending]}]')
                # find_option4gate(pending)
            if found:
                break
        # END: for loop that picks an edge
        return (weight, u, v) if found else ()

    def find_option4gate(gate):
        debug and print(f'<find_option4gate> starting... gate = '
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
        # () check gate crossings
        choice = first_non_crossing(choices, gate)
        if choice:
            # merging is better than gate, submit entry to pq
            weight, u, v = choice
            # tradeoff calculation
            tradeoff = weight - d2roots[gate, A.nodes[gate]['root']]
            pq.add(tradeoff, gate, (u, v))
            ComponIn[Gate[v]].add(gate)
            debug and print(f'<pushed> g2drop <{F[gate]}>, '
                            f'«{F[u]}–{F[v]}», tradeoff = {tradeoff:.1e}')
        else:
            # no viable edge is better than gate for this node
            # this becomes a final gate
            if i:  # run only if not at i = 0
                # definitive gates at iteration 0 do not cross any other edges
                # they are not included in Final_G because the algorithm
                # considers the gates extending to infinity (not really)
                root = A.nodes[gate]['root']
                make_gate_final(root, gate)
                check_heap4crossings(root, gate)
            debug and print('<cancelling>', F[gate])
            if gate in pq.tags:
                # i=0 gates and check_heap4crossings reverse_entry
                # may leave accepting gates out of pq
                pq.cancel(gate)

    def check_heap4crossings(root, finalG):
        '''search the heap for edges that cross the gate 'finalG'.
        calls find_option4gate for each of the subtrees involved'''
        for tradeoff, _, g2drop, uv in pq:
            # if uv is None or uv not in A.edges:
            if uv is None:
                continue
            u, v = uv
            if is_crossing_gate(root, finalG, u, v):
                nonlocal prevented_crossings
                # crossing gate, discard edge
                prevented_crossings += 1
                ban_queued_edge(g2drop, u, v)

    def ban_queued_edge(g2drop, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            debug and print('<<<< UNLIKELY <ban_queued_edge()> '
                            f'({F[u]}, {F[v]}) not in A.edges >>>>')
        g2keep = Gate[v]
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
            print(f'«{F[u]}–{F[v]}», '
                  f'g2drop <{F[g2drop]}>, g2keep <{F[g2keep]}> '
                  f'componin: {componin}, is_reverse: {is_reverse}')

        # END: block to be simplified

    # TODO: check if this function is necessary (not used)
    def abort_edge_addition(g2drop, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            print('<<<< UNLIKELY <abort_edge_addition()> '
                  f'({F[u]}, {F[v]}) not in A.edges >>>>')
        ComponIn[Gate[v]].remove(g2drop)
        find_option4gate(g2drop)

    # initialize pq
    for n in range(T):
        find_option4gate(n)

    log = []
    G.graph['log'] = log
    loop = True
    # BEGIN: main loop
    while loop:
        i += 1
        if i > maxiter:
            print(f'ERROR: maxiter reached ({i})')
            break
        debug and print(f'[{i}]')
        # debug and print(f'[{i}] bj–bm root: {A.edges[(F.bj, F.bm)]["root"]}')
        if gates2upd8:
            debug and print('gates2upd8:', ', '.join(F[gate] for gate in
                                                     gates2upd8))
        while gates2upd8:
            # find_option4gate(gates2upd8.popleft())
            find_option4gate(gates2upd8.pop())
        if not pq:
            # finished
            break
        g2drop, (u, v) = pq.top()
        debug and print(f'<popped> «{F[u]}–{F[v]}»,'
                        f' g2drop: <{F[g2drop]}>')

        # TODO: main loop should do only
        # - pop from pq
        # - check if adding edge would block some component
        # - add edge
        # - call find_option4gate for everyone affected

        # check if (u, v) crosses an existing edge
        if delaunay_based:
            # look for crossing edges within the neighborhood of (u, v)
            # faster, but only works if using the expanded delaunay edges
            eX = edge_crossings(u, v, G, diagonals)
        else:
            # when using the edges of a complete graph
            # alternate way - slower
            eX = []
            eXtmp = []
            eXnodes = set()
            nodes2check = set()
            uC, vC = VertexC[[u, v]]
            for s, t in G.edges:
                if s == u or t == u or s == v or t == v or s < 0 or t < 0:
                    # skip if the edges have a common node or (s, t) is a gate
                    continue
                if is_crossing(uC, vC, *VertexC[[s, t]], touch_is_cross=True):
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
                        if w != t and not is_same_side(uC, vC,
                                                       *VertexC[[w, t]]):
                            eX.append((s, t))
                elif t in nodes2check:
                    for w in G[t]:
                        if w != s and not is_same_side(uC, vC,
                                                       *VertexC[[w, s]]):
                            eX.append((s, t))
                else:
                    eX.append((s, t))

        if eX:
            print(f'<edge_crossing> discarding {(F[u], F[v])}: would cross'
                  f' {[(F[s], F[t]) for s, t in eX]}')
            # abort_edge_addition(g2drop, u, v)
            prevented_crossings += 1
            ban_queued_edge(g2drop, u, v)
            continue

        g2keep = Gate[v]
        root = A.nodes[g2keep]['root']

        capacity_left = capacity - len(subtrees[u]) - len(subtrees[v])

        # assess the union's angle span
        keepHi = ComponHiLim[g2keep]
        keepLo = ComponLoLim[g2keep]
        dropHi = ComponHiLim[g2drop]
        dropLo = ComponLoLim[g2drop]
        newHi = dropHi if angle(*VertexC[[keepHi, root, dropHi]]) > 0 else keepHi
        newLo = dropLo if angle(*VertexC[[dropLo, root, keepLo]]) > 0 else keepLo
        debug and print(f'<angle_span> //{F[newLo]} : '
                        f'{F[newHi]}//')

        # check which gates are within the union's angle span
        lR = anglesRank[newLo, root]
        hR = anglesRank[newHi, root]
        anglesWrap = lR > hR
        abort = False
        # the more conservative check would be using g2keep instead of
        # g2drop in the line below (but then the filter needs changing)
        distanceThreshold = d2rootsRank[g2drop, root]
        for gate in [g for g in G[root] if
                     d2rootsRank[g, root] > distanceThreshold]:
            gaterank = anglesRank[gate, root]
            if (not anglesWrap and (lR < gaterank < hR) or
                    (anglesWrap and (gaterank > lR or gaterank < hR))):
                # possible occlusion of subtree[gate] by union subtree
                debug and print(f'<check_occlusion> «{F[u]}-{F[v]}» might '
                                f'cross gate <{F[gate]}>')
                if gate in Final_G[root]:
                    if is_crossing_gate(root, gate, u, v, touch_is_cross=True):
                        abort = True
                        break
                elif gate in ComponIn[g2drop] or gate in ComponIn[g2keep]:
                    if (len(subtrees[gate]) > capacity_left):
                        # check crossing with gate
                        if is_crossing_gate(root, gate, u, v,
                                            touch_is_cross=True):
                            # find_option for gate, but forbidding g2drop, g2keep
                            abort = True
                            break
                    else:
                        debug and print(f'$$$$$ UNLIKELY: gate <{F[gate]}> '
                                        'could merge with subtree '
                                        f'<{F[g2keep]}> $$$$$')
                else:
                    # check crossing with next union for gate
                    entry = pq.tags.get(gate)
                    if entry is not None:
                        _, _, _, (s, t) = entry
                        if is_crossing(*VertexC[[u, v, s, t]],
                                       touch_is_cross=False):
                            abort = True
                            break

        if abort:
            debug and print(f'######## «{F[u]}-{F[v]}» would block '
                            f'gate {F[gate]} ########')
            prevented_crossings += 1
            ban_queued_edge(g2drop, u, v)
            continue

        # edge addition starts here
        subtree = subtrees[v]
        subtree |= subtrees[u]
        G.remove_edge(A.nodes[u]['root'], g2drop)
        log.append((i, 'remE', (A.nodes[u]['root'], g2drop)))

        g2keep_entry = pq.tags.get(g2keep)
        if g2keep_entry is not None:
            _, _, _, (_, t) = g2keep_entry
            # print('node', F[t], 'gate', F[Gate[t]])
            ComponIn[Gate[t]].remove(g2keep)
        # TODO: think about why a discard was needed
        ComponIn[g2keep].discard(g2drop)

        # update the component's angle span
        ComponHiLim[g2keep] = newHi
        ComponLoLim[g2keep] = newLo

        # assign root, gate and subtree to the newly added nodes
        for n in subtrees[u]:
            A.nodes[n]['root'] = root
            Gate[n] = g2keep
            subtrees[n] = subtree
        debug and print(f'<add edge> «{F[u]}-{F[v]}» gate '
                        f'<{F[g2keep]}>, '
                        f'heap top: <{F[pq[0][-2]]}>, '
                        f'«{chr(8211).join([F[x] for x in pq[0][-1]])}»'
                        f' {pq[0][0]:.1e}' if pq else 'heap EMPTY')
        #  G.add_edge(u, v, **A.edges[u, v])
        G.add_edge(u, v, **{weight_attr: A[u][v][weight_attr]})
        log.append((i, 'addE', (u, v)))
        # remove from consideration edges internal to subtrees
        A.remove_edge(u, v)

        # finished adding the edge, now check the consequences
        if capacity_left > 0:
            for gate in list(ComponIn[g2keep]):
                if len(subtrees[gate]) > capacity_left:
                    # TODO: think about why a discard was needed
                    # ComponIn[g2keep].remove(gate)
                    ComponIn[g2keep].discard(gate)
                    # find_option4gate(gate)
                    # gates2upd8.append(gate)
                    gates2upd8.add(gate)
            for gate in ComponIn[g2drop] - ComponIn[g2keep]:
                if len(subtrees[gate]) > capacity_left:
                    # find_option4gate(gate)
                    # gates2upd8.append(gate)
                    gates2upd8.add(gate)
                else:
                    ComponIn[g2keep].add(gate)
            # ComponIn[g2drop] = None
            # find_option4gate(g2keep)
            # gates2upd8.append(g2keep)
            gates2upd8.add(g2keep)
        else:
            # max capacity reached: subtree full
            if g2keep in pq.tags:  # if required because of i=0 gates
                pq.cancel(g2keep)
            make_gate_final(root, g2keep)
            # don't consider connecting to this full subtree nodes anymore
            A.remove_nodes_from(subtree)
            for gate in ComponIn[g2drop] | ComponIn[g2keep]:
                # find_option4gate(gate)
                # gates2upd8.append(gate)
                gates2upd8.add(gate)
            # ComponIn[g2drop] = None
            # ComponIn[g2keep] = None
            check_heap4crossings(root, g2keep)
    # END: main loop

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
    G.graph['iterations'] = i
    G.graph['prevented_crossings'] = prevented_crossings
    G.graph['capacity'] = capacity
    G.graph['creator'] = 'CPEW'
    G.graph['edges_fun'] = CPEW
    G.graph['creation_options'] = options
    G.graph['runtime_unit'] = 's'
    G.graph['runtime'] = time.perf_counter() - start_time
    return G
