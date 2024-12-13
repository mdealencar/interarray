# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import networkx as nx

from .crossings import list_edge_crossings
from .interarraylib import calcload, NodeTagger
from . import warn


F = NodeTagger()


def gate_and_leaf_path(S: nx.Graph, n: int) -> tuple[int, int]:
    '''
    `S` has loads, is a rootless subgraph_view and non-branching
    '''
    # non-branching graphs only, gates and detours removed
    if S.degree[n] == 2:
        u, v = S[n]
        head, tail = ((u, v)
                      if S.nodes[u]['load'] > S.nodes[v]['load'] else
                      (v, u))
        # go towards the gate
        gate_leaf = []
        for fwd, back in ((head, tail), (tail, head)):
            while S.degree[fwd] == 2:
                s, t = S[fwd]
                fwd, back = (s, fwd) if t == back else (t, fwd)
            gate_leaf.append(fwd)
        return tuple(gate_leaf)
    else:
        if S.nodes[n]['load'] == 1:
            leaf = back = n
            gate = None
        else:
            gate = back = n
        fwd, = S[n]
        while S.degree[fwd] == 2:
            s, t = S[fwd]
            fwd, back = (s, fwd) if t == back else (t, fwd)
        if gate is None:
            gate_leaf = (fwd, leaf)
        else:
            gate_leaf = (gate, fwd)
        return gate_leaf


def list_path(S: nx.Graph, n: int) -> list[int]:
    '''
    `S` has loads, no gate or detour edges
    all subtrees of `S` are paths
    `n` must be an extremity of the path
    '''
    path = [n]
    far, = S[n]
    while S.degree[far] == 2:
        s, t = S[far]
        far, n = (s, far) if t == n else (t, far)
        path.append(n)
    path.append(far)
    return path if S.nodes[far]['load'] == 1 else path[::-1]


def _find_fix_choices_path(A: nx.Graph, swapS: int, src_path: list[int],
                           dst_path: list[int]) -> list[tuple]:
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


def _quantify_choices(S, A, swapS, src_path, dst_path, choices):
    quant_choices = []
    rootS, = (n for n in S[src_path[0]] if n < 0)
    rootD, = (n for n in S[dst_path[0]] if n < 0)
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
            gates_del.append((rootS, swapS))
            change -= d2roots[swapS, rootS]
            gates_add.append((rootS, gateS))
            change += minSd2root
        elif gateS != src_path[0]:
            gates_del.append((rootS, src_path[0]))
            change -= d2roots[src_path[0], rootS]
            gates_add.append((rootS, gateS))
            change += minSd2root
        if gateD != dst_path[0]:
            gates_del.append((rootD, dst_path[0]))
            change -= d2roots[dst_path[0], rootD]
            minDd2root, gateD = min(
                (d2roots[gateD, rootD], gateD),
                (d2roots[dst_path[-1], rootD], dst_path[-1]),
            )
            gates_add.append((rootD, gateD))
            change += minDd2root
        change += (sum(A[u][v]['length'] for u, v in edges_add)
                   - sum(A[u][v]['length'] for u, v in edges_del))
        quant_choices.append((change, (edges_del, edges_add, gates_del,
                                       gates_add)))
    return quant_choices


def _apply_choice(S: nx.Graph, A: nx.Graph, edges_del: list[tuple[int, int]],
        edges_add: list[tuple[int, int]], gates_del: list[tuple[int, int]],
        gates_add: list[tuple[int, int]]) -> nx.Graph:
    d2roots = A.graph['d2roots']
    # for edges: add first, then del
    S.add_edges_from(edges_add)
    S.remove_edges_from(edges_del)
    # for gates: del first, then add
    for root, gate in gates_del:
        S.remove_edge(root, gate)
    for root, gate in gates_add:
        S.add_edge(root, gate, length=d2roots[gate, root])
    # the repair invalidates current load attributes -> recalculate them
    # TODO: do it in a smarter way by updating only affected subtrees
    calcload(S)
    return S


def repair_routeset_path(Sʹ: nx.Graph, A: nx.Graph) -> nx.Graph:
    # naming: suffix _path as opposed to _tree -> Sʹ is non-branching
    '''

    This is only able to repair crossings where one of the paths is split in
    either a leaf and a path or a hook and a path.

    Args:
        Sʹ: solution topology that contains non-branching rooted tree(s)
        A: available edges used in creating `Sʹ`

    Returns:
        Topology without the crossing in a shallow copy of `Sʹ`.
    '''

    if 'C' in Sʹ.graph or 'D' in Sʹ.graph:
        print('ERROR: no changes made - `repair_routeset_path()` requires '
              '`Sʹ` as a topology.')
        return Sʹ
    P = A.graph['planar']
    diagonals = A.graph['diagonals']
    S = Sʹ.copy()

    def not_crossing(choice):
        # reads from parent scope: diagonals, S, P
        #  gateD, swapD, freeS, edges_del, edges_add = choice
        _, _, _, edges_del, edges_add = choice
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
                if (((s, t) in S.edges
                        or (s, t) in edges_add)
                        and (s, t) not in edges_del):
                    # crossing with diagonal
                    return False
            else:
                # ⟨u, v⟩ is a diagonal of Delaunay ⟨s, t⟩
                s, t = st
                if (((s, t) in S.edges
                        or (s, t) in edges_add)
                        and (s, t) not in edges_del):
                    # crossing with Delaunay edge
                    return False

                # TODO: update the code below to use the bidict diagonals
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
                            and (diag_da in S.edges
                                 or diag_da in edges_add)
                            and diag_da not in edges_del):
                        return False
                    e = P[a][c]['ccw']
                    diag_eb = (e, b) if e < b else (b, e)
                    if (e == P[c][a]['cw']
                            and (diag_eb in S.edges
                                 or diag_eb in edges_add)
                            and diag_eb not in edges_del):
                        return False
        return True

    outstanding_crossings = []
    while True:
        # make a subgraph without gates and detours
        S_T = nx.subgraph_view(S, filter_node=lambda n: n >= 0)
        eeXings = list_edge_crossings(S_T, A)
        done = True
        for uv, st in eeXings:
            uv, st = (uv, st) if uv < st else (st, uv)
            if (uv, st) in outstanding_crossings:
                continue
            if all(S_T.degree[n] > 1 for n in (*uv, *st)):
                # found an unrepairable crossing
                outstanding_crossings.append((uv, st))
                continue
            # crossing is potentialy repairable
            done = False
            u, v = uv
            s, t = st
            break
        if done:
            break
        gateV, leafV = gate_and_leaf_path(S_T, v)
        gateT, leafT = gate_and_leaf_path(S_T, t)
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
            warn('Crossing repair is only implemented for the cases where the '
                 'split results in at least one single-noded branch fragment.')
            outstanding_crossings.append((uv, st))
            continue
        quant_choices = []
        for gateS, gateD, swapS in src_dst_swap:
            src_path = list_path(S_T, gateS)
            dst_path = list_path(S_T, gateD)
            # TODO: remove this
            #  print([f'{F[n]}' for n in dst_path])
            choices = _find_fix_choices_path(A, swapS, src_path, dst_path)
            choices = filter(not_crossing, choices)
            quant_choices.extend(
                _quantify_choices(S, A, swapS, src_path, dst_path, choices)
            )
        if not quant_choices:
            warn('Unrepairable: no suitable node swap found.')
            outstanding_crossings.append((uv, st))
            continue
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
        # apply_choice works on a copy of Sʹ
        S = _apply_choice(S, A, *choice)
        S.graph['repaired'] = 'repair_routeset_path'
    if outstanding_crossings:
        S.graph['num_crossings'] = len(outstanding_crossings)
        S.graph['outstanding_crossings'] = outstanding_crossings
    return S
