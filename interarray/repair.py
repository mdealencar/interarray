# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import networkx as nx
from scipy.spatial.distance import cdist

from .pathfinding import PathFinder
from .mesh import make_planar_embedding
from .crossings import list_edge_crossings
from .interarraylib import remove_detours, calcload, NodeTagger


F = NodeTagger()


def gate_and_leaf_path(G: nx.Graph, n: int) -> tuple[int, int]:
    '''
    `G` has no gate or detour edges and is non-branching
    '''
    # non-branching graphs only, gates and detours removed
    if G.degree(n) == 2:
        u, v = G[n]
        head, tail = ((u, v)
                      if G.nodes[u]['load'] > G.nodes[v]['load'] else
                      (v, u))
        # go towards the gate
        gate_leaf = []
        for fwd, back in ((head, tail), (tail, head)):
            while G.degree(fwd) == 2:
                s, t = G[fwd]
                fwd, back = (s, fwd) if t == back else (t, fwd)
            gate_leaf.append(fwd)
        return tuple(gate_leaf)
    else:
        if G.nodes[n]['load'] == 1:
            leaf = back = n
            gate = None
        else:
            gate = back = n
        fwd, = G[n]
        while G.degree(fwd) == 2:
            s, t = G[fwd]
            fwd, back = (s, fwd) if t == back else (t, fwd)
        if gate is None:
            gate_leaf = (fwd, leaf)
        else:
            gate_leaf = (gate, fwd)
        return gate_leaf


def list_path(G: nx.graph, n: int) -> list[int]:
    '''
    `G` has no gate or detour edges
    all subtrees of `G` are paths
    `n` must be an extremity of the path
    '''
    path = [n]
    far, = G[n]
    while G.degree(far) == 2:
        s, t = G[far]
        far, n = (s, far) if t == n else (t, far)
        path.append(n)
    path.append(far)
    return path if G.nodes[far]['load'] == 1 else path[::-1]


def _find_fix_choices_path(A: nx.Graph, swapS: int, src_path: list[int],
                           dst_path: list[int]) -> tuple[tuple]:
    # (G, A, swapS, hookS_alt, dst_gate):
    # this is named «...»_path because we could make a version that allows
    # branching and call it «...»_tree.
    '''
    Swap node `swapS` with one of the nodes of `dst_path`. For each swap, try
    all possible point of insertion of `swapS` into `dst_path`.

    how it works:
    - Several node swapping choices are examined within the edges in `A`.
    - A list where each item is a feasible modification package is returned.
    '''
    i_end = len(dst_path)
    choices = []
    hookS_cut, hookS_alt = (
        (src_path[1], src_path[-1])
        if swapS == src_path[0] else
        (src_path[-2], src_path[0])
    )
    gateD = dst_path[0]
    edges_del_0 = [(swapS, hookS_cut)]
    for hookS, freeS in ((hookS_cut, hookS_alt),
                         (hookS_alt, hookS_cut)):
        for i, swapD in enumerate(dst_path):
            # loop through nodes of dst_path (to remove)
            if swapD not in A[hookS]:
                # no easy way to link swapD to src path
                continue
            edges_add_0 = [(swapD, hookS)]
            # three cases here regarding dst_path:
            # A) bypass swapD (insert swapS anywhere in dst_path)
            # B) insert swapS in swapD's place (cannot bypass swapD)
            # C) gate or leaf of dst_path as swapD (insert swapS anywhere)
            # D) none of the above is possible
            if (0 < i < i_end - 1):  # mid-path swapD (remove)
                nearD_, farD_ = dst_path[i - 1], dst_path[i + 1]
                edges_del_1 = [(nearD_, swapD), (swapD, farD_)]
                if nearD_ in A[farD_]:
                    # bypassing of swapD is possible
                    edges_add_1 = [(nearD_, farD_)]
                    dst_path_ = dst_path[:i] + dst_path[i + 1:]
                    # case (A)
                    for nearD, farD in zip(dst_path_[:-1], dst_path_[1:]):
                        # loop through mid-positions in dst_path_ (insert)
                        if nearD in A[swapS] and farD in A[swapS]:
                            # fix found
                            edges_del = (edges_del_0 + edges_del_1
                                         + [(nearD, farD)])
                            edges_add = (edges_add_0 + edges_add_1
                                         + [(nearD, swapS),
                                            (swapS, farD)])
                            choices.append((gateD, swapD, freeS, edges_del,
                                            edges_add))
                    for hookD, D_gated in ((dst_path_[0], False),
                                           (dst_path_[-1], True)):
                        # loop through extreme-positions in dst_path_ (extend)
                        if hookD in A[swapS]:
                            # fix found
                            edges_del = edges_del_0 + edges_del_1
                            edges_add = (edges_add_0 + edges_add_1
                                         + [(hookD, swapS)])
                            choices.append((gateD if D_gated else swapS, swapD,
                                            freeS, edges_del, edges_add))
                    if nearD_ in A[swapS] and farD_ in A[swapS]:
                        # case (B) – single insertion position in dst
                        # fix found
                        edges_del = edges_del_0 + edges_del_1
                        edges_add = (edges_add_0 +
                                     [(nearD_, swapS),
                                      (swapS, farD_)])
                        choices.append((gateD, swapD, freeS, edges_del,
                                        edges_add))
                    else:
                        # case (D) – nothing to do
                        continue
            else:
                # case (C)
                dst_path_, hookD, D_gated = (
                    (dst_path[1:], dst_path[1], False)
                    if i == 0 else
                    (dst_path[:-1], dst_path[-2], True)
                )
                edges_del_1 = [(swapD, hookD)]
                for nearD, farD in zip(dst_path_[:-1], dst_path_[1:]):
                    # loop through mid-positions in dst_path_ (to insert)
                    if nearD in A[swapS] and farD in A[swapS]:
                        # fix found
                        edges_del = (edges_del_0 + edges_del_1
                                     + [(nearD, farD)])
                        edges_add = (edges_add_0
                                     + [(nearD, swapS),
                                        (swapS, farD)])
                        choices.append((gateD if D_gated else hookD,
                                        swapD, freeS, edges_del, edges_add))
                if hookD in A[swapS]:
                    # fix found
                    edges_del = edges_del_0 + edges_del_1
                    edges_add = (edges_add_0 + [(hookD, swapS)])
                    choices.append((gateD if D_gated else swapS,
                                    swapD, freeS, edges_del, edges_add))
    return choices  # ((gateD, swapD, freeS, edges_del, edges_add), ...)


def _quantify_choices(G, A, swapS, src_path, dst_path, choices):
    quant_choices = []
    rootS, = (n for n in G[src_path[0]] if n < 0)
    rootD, = (n for n in G[dst_path[0]] if n < 0)
    d2roots = G.graph.get('d2roots')
    if d2roots is None:
        VertexC = G.graph['VertexC']
        M = G.graph['M']
        d2roots = cdist(VertexC[:-M], VertexC[-M:])
        G.graph['d2roots'] = d2roots
    for gateD, swapD, freeS, edges_del, edges_add in choices:
        gates_del = []
        gates_add = []
        change = 0
        minSd2root, gateS = min(
            (d2roots[swapD, rootS], swapD),
            (d2roots[freeS, rootS], freeS),
        )
        if swapS == src_path[0]:
            gates_del.append((swapS, rootS))
            change -= d2roots[swapS, rootS]
            gates_add.append((gateS, rootS))
            change += minSd2root
        elif gateS != src_path[0]:
            gates_del.append((src_path[0], rootS))
            change -= d2roots[src_path[0], rootS]
            gates_add.append((gateS, rootS))
            change += minSd2root
        if gateD != dst_path[0]:
            gates_del.append((dst_path[0], rootD))
            change -= d2roots[dst_path[0], rootD]
            minDd2root, gateD = min(
                (d2roots[gateD, rootD], gateD),
                (d2roots[dst_path[-1], rootD], dst_path[-1]),
            )
            gates_add.append((gateD, rootD))
            change += minDd2root
        change += (sum(A[u][v]['length'] for u, v in edges_add)
                   - sum(A[u][v]['length'] for u, v in edges_del))
        quant_choices.append((change, (swapD, edges_del, edges_add, gates_del,
                                       gates_add)))
    return quant_choices


def _apply_choice(
        G_orig: nx.Graph, A: nx.Graph, swapS: int, swapD: int,
        edges_del: list[tuple[int, int]], edges_add: list[tuple[int, int]],
        gates_del: list[tuple[int, int]], gates_add: list[tuple[int, int]]
        ) -> nx.Graph:
    d2roots = G_orig.graph['d2roots']
    G = remove_detours(G_orig)
    # for edges: add first, then del
    G.add_weighted_edges_from(((u, v, A[u][v]['length'])
                               for u, v in edges_add), weight='length')
    G.remove_edges_from(edges_del)
    # for gates: del first, then add
    for gate, root in gates_del:
        G.remove_edge(gate, root)
    for gate, root in gates_add:
        G.add_edge(gate, root, length=d2roots[gate, root])
    calcload(G)
    return G


def repair_routeset_path(G: nx.Graph, A: nx.Graph) -> nx.Graph:
    '''
    `G` is a NetworkX.Graph representing a routeset that is:
        - topologically sound (a tree, respecting capacity)
        - non-branching

    This is only able to repair crossings where one of the paths is crossing
    on one of its extreme edges (non-gate/detour).

    Returns:
        - a fixed routeset in a shallow copy of `G`.
    '''

    M, N, B, C, D = (G.graph.get(k, 0) for k in ('M', 'N', 'B', 'C', 'D'))

    P = A.graph['planar']
    diagonals = A.graph['diagonals']

    G2fix = remove_detours(G) if D else G.copy()
    # make a subgraph without gates and detours
    # TODO: make G_branches compatible with contour edges
    G_branches = nx.subgraph_view(G2fix, filter_node=lambda n: n >= 0)
    eeXings = list_edge_crossings(G_branches, P, diagonals)
    if eeXings:
        G2fix.graph['crossings_fixed'] = 0

    def not_crossing(choice):
        gateD, swapD, freeS, edges_del, edges_add = choice
        edges_del_ = set((u, v) if u < v else (v, u) for u, v in edges_del)
        edges_add_ = set((u, v) if u < v else (v, u) for u, v in edges_add)
        edges_add = edges_add_ - edges_del_
        edges_del = edges_del_ - edges_add_
        for u, v in edges_add:
            st = diagonals.get((u, v))
            if st is None:
                # ⟨u, v⟩ is a Delaunay edge, find its diagonal
                st = diagonals.inv.get((u, v))
                if st is None:
                    # ⟨u, v⟩ has no diagonal
                    continue
                s, t = st
                if s < 0 or t < 0:
                    # diagonal of ⟨u, v⟩ is a gate
                    continue
                if (((s, t) in G2fix.edges
                        or (s, t) in edges_add)
                        and (s, t) not in edges_del):
                    # crossing with diagonal
                    return False
            else:
                # ⟨u, v⟩ is a diagonal of Delaunay ⟨s, t⟩
                s, t = st
                if (((s, t) in G2fix.edges
                        or (s, t) in edges_add)
                        and (s, t) not in edges_del):
                    # crossing with Delaunay edge
                    return False

                # ensure u–s–v–t is ccw
                u, v = ((u, v)
                        if (P[u][t]['cw'] == s and P[v][s]['cw'] == t) else
                        (v, u))
                # examine the two triangles ⟨s, t⟩ belongs to
                for a, b, c in ((s, t, u), (t, s, v)):
                    # this is for diagonals crossing diagonals (4 checks)
                    d = P[c][b]['ccw']
                    diag_da = (a, d) if a < d else (d, a)
                    if (d == P[b][c]['cw']
                            and (diag_da in G2fix.edges
                                 or diag_da in edges_add)
                            and diag_da not in edges_del):
                        return False
                    e = P[a][c]['ccw']
                    diag_eb = (e, b) if e < b else (b, e)
                    if (e == P[c][a]['cw']
                            and (diag_eb in G2fix.edges
                                 or diag_eb in edges_add)
                            and diag_eb not in edges_del):
                        return False
        return True

    while eeXings:
        if 'has_loads' not in G.graph:
            calcloads(G2fix)
        (u, v), (s, t) = eeXings[0]
        gateV, leafV = gate_and_leaf_path(G_branches, v)
        gateT, leafT = gate_and_leaf_path(G_branches, t)
        src_dst_swap = []
        if u == gateV or u == leafV:
            src_dst_swap.append((gateV, gateT, u))
        elif v == gateV or v == leafV:
            src_dst_swap.append((gateV, gateT, v))
        if s == gateT or s == leafT:
            src_dst_swap.append((gateT, gateV, s))
        elif t == gateT or t == leafT:
            src_dst_swap.append((gateT, gateV, t))

        if not src_dst_swap:
            print('ERROR: unable to fix crossing that does not have a single '
                  'node in one of the cuts.')
            return G
        quant_choices = []
        for gateS, gateD, swapS in src_dst_swap:
            src_path = list_path(G_branches, gateS)
            dst_path = list_path(G_branches, gateD)
            # TODO: remove this
            print([f'{F[n]}' for n in dst_path])
            choices = _find_fix_choices_path(A, swapS, src_path, dst_path)
            choices = filter(not_crossing, choices)
            quant_choices.extend(
                _quantify_choices(G2fix, A, swapS, src_path, dst_path, choices)
            )
        if not quant_choices:
            print('ERROR: unable to find fix.')
            return G
        quant_choices.sort()
        # TODO: remove this
        #  for line in [' | '.join(
        #      (f'{cost:.3f}', F[swapD],
        #          ' '.join(['–'.join((F[u], F[v])) for u, v in edges_del]),
        #          ' '.join(['–'.join((F[u], F[v])) for u, v in edges_add]),
        #          ' '.join(['–'.join((F[u], F[v])) for u, v in gates_del]),
        #          ' '.join(['–'.join((F[u], F[v])) for u, v in gates_add])))
        #          for cost, (swapD, edges_del, edges_add, gates_del,
        #                     gates_add)
        #          in quant_choices]:
        #      print(line)
        _, choice = quant_choices[0]
        # apply_choice works on a copy of G
        G2fix = _apply_choice(G2fix, A, swapS, *choice)
        G2fix.graph['crossings_fixed'] += 1
        del G2fix.graph['has_loads']
        G_branches = nx.subgraph_view(G2fix, filter_node=lambda n: 0 <= n < N)
        eeXings = list_edge_crossings(G_branches, P, diagonals)

    #  PathFinder(G2fix, branching=False).create_detours(in_place=True)

    return G2fix
