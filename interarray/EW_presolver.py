# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import time

import numpy as np
import networkx as nx
from scipy.stats import rankdata

from .geometric import angle, assign_root
from .crossings import edge_crossings
from .utils import NodeTagger
from .priorityqueue import PriorityQueue
from .interarraylib import calcload

F = NodeTagger()


def EW_presolver(Aʹ: nx.Graph, capacity: int, maxiter=10000, debug=False) -> nx.Graph:
    '''Modified Esau-Williams heuristic for C-MST with limited crossings
    
    Args:
        A: available edges graph
        capacity: maximum link capacity
        maxiter: fail-safe to avoid locking in an infinite loop

    Returns:
        Solution topology.
    '''

    start_time = time.perf_counter()
    R, T = (Aʹ.graph[k] for k in 'RT')
    diagonals = Aʹ.graph['diagonals']
    d2roots = Aʹ.graph['d2roots']
    S = nx.Graph(
        R=R, T=T,
        capacity=capacity,
        handle=Aʹ.graph['handle'],
        creator='EW_presolver',
        edges_fun=EW_presolver,
        creation_options={},
    )
    A = Aʹ.copy()

    roots = range(-R, 0)
    VertexC = A.graph['VertexC']

    assign_root(A)
    d2rootsRank = rankdata(d2roots, method='dense', axis=0)

    # removing root nodes from A to speedup find_option4gate
    # this may be done because G already starts with gates
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # BEGIN: create initial star graph
    S.add_edges_from(((n, r) for n, r in A.nodes(data='root')))
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

    def component_merging_edge(gate, forbidden=None, margin=1.02):
        # gather all the edges leaving the subtree of gate
        if forbidden is None:
            forbidden = set()
        forbidden.add(gate)
        capacity_left = capacity - len(subtrees[gate])
        choices = []
        gate_d2root = d2roots[gate, A.nodes[gate]['root']]
        #  weighted_edges = []
        edges2discard = []
        for u in subtrees[gate]:
            for v in A[u]:
                if (Gate[v] in forbidden or
                        len(subtrees[v]) > capacity_left):
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    W = A[u][v]['length']
                    # DEVIATION FROM Esau-Williams: slack
                    if W <= gate_d2root:
                        # useful edges
                        # v's proximity to root is used as tie-breaker
                        choices.append(
                            (W, d2rootsRank[v, A.nodes[v]['root']], u, v))
        if not choices:
            return None, 0., edges2discard
        choices.sort()
        best_W, best_rank, *best_edge = choices[0]
        for W, rank, *edge in choices[1:]:
            if W > margin*best_W:
                # no more edges within margin
                break
            if  rank < best_rank:
                best_W, best_rank, best_edge = W, rank, edge
        tradeoff = best_W - gate_d2root
        return best_edge, tradeoff, edges2discard

    def find_option4gate(gate):
        debug and print(f'<find_option4gate> starting... gate = '
                        f'<{F[gate]}>')
        if edges2ban:
            debug and print(f'<<<<<<<edges2ban>>>>>>>>>>> _{len(edges2ban)}_')
        while edges2ban:
            # edge2ban = edges2ban.popleft()
            edge2ban = edges2ban.pop()
            ban_queued_edge(*edge2ban)
        # () get component expansion edge with weight
        edge, tradeoff, edges2discard = component_merging_edge(gate)
        # discard useless edges
        A.remove_edges_from(edges2discard)
        if edge is not None:
            # merging is better than gate, submit entry to pq
            # tradeoff calculation
            pq.add(tradeoff, gate, edge)
            ComponIn[Gate[edge[1]]].add(gate)
            debug and print(f'<pushed> g2drop <{F[gate]}>, '
                            f'«{F[edge[0]]}–{F[edge[1]]}», tradeoff = {tradeoff:.1e}')
        else:
            # no viable edge is better than gate for this node
            debug and print('<cancelling>', F[gate])
            if gate in pq.tags:
                pq.cancel(gate)

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

    # initialize pq
    for n in range(T):
        find_option4gate(n)

    log = []
    S.graph['log'] = log
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
        # look for crossing edges within the neighborhood of (u, v)
        # this works for expanded delaunay edges (see CPEW for all edges)
        eX = edge_crossings(u, v, S, diagonals)

        if eX:
            debug and print(f'<edge_crossing> discarding {(F[u], F[v])}: would cross'
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

        # edge addition starts here
        subtree = subtrees[v]
        subtree |= subtrees[u]
        S.remove_edge(A.nodes[u]['root'], g2drop)
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
        S.add_edge(u, v)
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
            # don't consider connecting to this full subtree nodes anymore
            A.remove_nodes_from(subtree)
            for gate in ComponIn[g2drop] | ComponIn[g2keep]:
                # find_option4gate(gate)
                # gates2upd8.append(gate)
                gates2upd8.add(gate)
            # ComponIn[g2drop] = None
            # ComponIn[g2keep] = None
    # END: main loop

    calcload(S)
    # algorithm finished, store some info in the graph object
    S.graph.update(
        iterations=i,
        prevented_crossings=prevented_crossings,
        runtime=time.perf_counter() - start_time,
    )
    return S
