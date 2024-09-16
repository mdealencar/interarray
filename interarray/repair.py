# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import networkx as nx
from scipy.spatial.distance import cdist

from .pathfinding import PathFinder
from .mesh import make_planar_embedding
from .crossings import list_edge_crossings
from .interarraylib import calcload, NodeTagger


F = NodeTagger()


def gate_and_leaf_path(T: nx.Graph, n: int) -> tuple[int, int]:
    '''
    `T` has loads, is a rootless subgraph_view and non-branching
    '''
    # non-branching graphs only, gates and detours removed
    if T.degree(n) == 2:
        u, v = T[n]
        head, tail = ((u, v)
                      if T.nodes[u]['load'] > T.nodes[v]['load'] else
                      (v, u))
        # go towards the gate
        gate_leaf = []
        for fwd, back in ((head, tail), (tail, head)):
            while T.degree(fwd) == 2:
                s, t = T[fwd]
                fwd, back = (s, fwd) if t == back else (t, fwd)
            gate_leaf.append(fwd)
        return tuple(gate_leaf)
    else:
        if T.nodes[n]['load'] == 1:
            leaf = back = n
            gate = None
        else:
            gate = back = n
        fwd, = T[n]
        while T.degree(fwd) == 2:
            s, t = T[fwd]
            fwd, back = (s, fwd) if t == back else (t, fwd)
        if gate is None:
            gate_leaf = (fwd, leaf)
        else:
            gate_leaf = (gate, fwd)
        return gate_leaf


def list_path(T: nx.graph, n: int) -> list[int]:
    '''
    `T` has loads, no gate or detour edges
    all subtrees of `T` are paths
    `n` must be an extremity of the path
    '''
    path = [n]
    far, = T[n]
    while T.degree(far) == 2:
        s, t = T[far]
        far, n = (s, far) if t == n else (t, far)
        path.append(n)
    path.append(far)
    return path if T.nodes[far]['load'] == 1 else path[::-1]


def _find_fix_choices_path(A: nx.Graph, swapS: int, src_path: list[int],
                           dst_path: list[int]) -> tuple[tuple]:
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


def _quantify_choices(T, A, swapS, src_path, dst_path, choices):
    quant_choices = []
    rootS, = (n for n in T[src_path[0]] if n < 0)
    rootD, = (n for n in T[dst_path[0]] if n < 0)
    d2roots = A.graph['d2roots']
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
        T: nx.Graph, A: nx.Graph, swapS: int, swapD: int,
        edges_del: list[tuple[int, int]], edges_add: list[tuple[int, int]],
        gates_del: list[tuple[int, int]], gates_add: list[tuple[int, int]]
        ) -> nx.Graph:
    d2roots = A.graph['d2roots']
    # for edges: add first, then del
    T.add_edges_from(edges_add)
    T.remove_edges_from(edges_del)
    # for gates: del first, then add
    for gate, root in gates_del:
        T.remove_edge(gate, root)
    for gate, root in gates_add:
        T.add_edge(gate, root, length=d2roots[gate, root], kind='tentative')
    if gates_add:
        tentative = T.graph.get(tentative)
        if tentative is None:
            T.graph['tentative'] = gates_add
        else:
            tentative.extend(gates_add)
    calcload(T)
    return T


def repair_routeset_path(T: nx.Graph, A: nx.Graph) -> nx.Graph:
    # naming: suffix _path as opposed to _tree -> T is non-branching
    '''
    `T` is a NetworkX.Graph representing a routeset that is:
        - topologically sound (a tree, respecting capacity)
        - non-branching

    This is only able to repair crossings where one of the paths is crossing
    on one of its extreme edges (non-gate/detour).

    Returns:
        - a routeset without the crossing in a shallow copy of `T`.
    '''

    if 'C' in T.graph or 'D' in T.graph:
        print('ERROR: `repair_routeset_path()` requires `T` as a topology.')
        return
    M, N = (T.graph[k] for k in 'MN')
    P = A.graph['planar']
    diagonals = A.graph['diagonals']
    T2fix = T.copy()

    # make a subgraph without gates and detours
    T_branches = nx.subgraph_view(T2fix, filter_node=lambda n: n >= 0)
    eeXings = list_edge_crossings(T_branches, A)
    if eeXings:
        T2fix.graph['crossings_fixed'] = 0

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
                if (((s, t) in T2fix.edges
                        or (s, t) in edges_add)
                        and (s, t) not in edges_del):
                    # crossing with diagonal
                    return False
            else:
                # ⟨u, v⟩ is a diagonal of Delaunay ⟨s, t⟩
                s, t = st
                if (((s, t) in T2fix.edges
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
                            and (diag_da in T2fix.edges
                                 or diag_da in edges_add)
                            and diag_da not in edges_del):
                        return False
                    e = P[a][c]['ccw']
                    diag_eb = (e, b) if e < b else (b, e)
                    if (e == P[c][a]['cw']
                            and (diag_eb in T2fix.edges
                                 or diag_eb in edges_add)
                            and diag_eb not in edges_del):
                        return False
        return True

    while eeXings:
        calcload(T2fix)
        (u, v), (s, t) = eeXings[0]
        gateV, leafV = gate_and_leaf_path(T_branches, v)
        gateT, leafT = gate_and_leaf_path(T_branches, t)
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
            print('ERROR: unable to fix crossing that has more than one '
                  'node in both components.')
            return T
        quant_choices = []
        for gateS, gateD, swapS in src_dst_swap:
            src_path = list_path(T_branches, gateS)
            dst_path = list_path(T_branches, gateD)
            # TODO: remove this
            print([f'{F[n]}' for n in dst_path])
            choices = _find_fix_choices_path(A, swapS, src_path, dst_path)
            choices = filter(not_crossing, choices)
            quant_choices.extend(
                _quantify_choices(T2fix, A, swapS, src_path, dst_path, choices)
            )
        if not quant_choices:
            print('ERROR: unable to find fix.')
            return T
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
        # apply_choice works on a copy of T
        T2fix = _apply_choice(T2fix, A, swapS, *choice)
        T2fix.graph['crossings_fixed'] += 1
        del T2fix.graph['has_loads']
        T_branches = nx.subgraph_view(T2fix, filter_node=lambda n: 0 <= n < N)
        eeXings = list_edge_crossings(T_branches, A)

    #  PathFinder(T2fix, branching=False).create_detours(in_place=True)

    return T2fix
