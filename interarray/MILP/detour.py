# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import numpy as np
from interarray.interarraylib import NodeTagger, NodeStr
from scipy.spatial.distance import cdist
from interarray.geometric import (
    angle, is_bunch_split_by_corner, is_crossing
)

F = NodeTagger()


def get_crossings(G, s, t, detour_waiver=False):
    '''generic crossings checker
    common node is not crossing'''
    M = G.graph['M']
    N = G.number_of_nodes() - M
    VertexC = G.graph['VertexC']
    fnT = np.arange(N + M)

    s_, t_ = fnT[[s, t]]
    # st = frozenset((s_, t_))
    # if st in triangles or st in triangles_exp:
    #     # <(s_, t_) is in the expanded Delaunay edge set>
    #     Xlist = edge_crossings(s_, t_, G, triangles, triangles_exp)
    #     # crossings with expanded Delaunay already checked
    #     # just detour edges missing
    #     nbunch = list(range(N, N + D))
    # else:
    # <(s, t) is not in the expanded Delaunay edge set>
    Xlist = []
    nbunch = None  # None means all nodes

    sC, tC = VertexC[[s_, t_]]
    # st_is_detour = s >= N or t >= N
    for w_x in G.edges(nbunch):
        w, x = w_x
        w_, x_ = fnT[[w, x]]
        # both_detours = st_is_detour and (w >= N or x >= N)
        skip = detour_waiver and (w >= N or x >= N)
        if skip or s_ == w_ or t_ == w_ or s_ == x_ or t_ == x_:
            # <edges have a common node>
            continue
        if is_crossing(sC, tC, *VertexC[[w_, x_]], touch_is_cross=True):
            Xlist.append(w_x)
    return Xlist


def get_barrier(Gattr, barrier_gate):
    root = Gattr['Root'][barrier_gate]
    Barrier = np.array(Gattr['Subtree'][barrier_gate], dtype=int)
    # get barrier extremities
    barrier_ranks = Gattr['anglesRank'][Barrier, root]
    barrierHi = Barrier[np.argmax(barrier_ranks)]
    barrierLo = Barrier[np.argmin(barrier_ranks)]

    Xhp = Gattr['anglesXhp'][[barrierHi, barrierLo], root]
    hiYhp, loYhp = Gattr['anglesYhp'][[barrierHi, barrierLo], root]

    if (not any(Xhp)) and hiYhp != loYhp:
        # extremities wrap across +-pi
        barrierHi, barrierLo = barrierLo, barrierHi
    return Barrier, barrierLo, barrierHi


def detour(G, root, org, dst, u, v, barrier_gate, depth=0,
           remove=set(), debug=False, maxDepth=5, savings=np.inf):
    # org (origin) is blocked
    # dst (destination) is goal_
    #(root, blocked, goal_, u, v, barrierLo,
    #barrierHi, savings, depth=0, remove=set()):
    # TODO: deprecate this in favor of interarray/pathfinding.py's PathFinder
    '''
    Deprecated. Use interarray/pathfinding.py's PathFinder instead

    This function was hastly adapted from ObstacleBypassingEW() to work on MILP-generated
    layouts. Kind of works, but be careful.
    '''

    M = G.graph['M']
    N = G.number_of_nodes() - M
    d2roots = G.graph['d2roots']
    d2rootsRank = G.graph['d2rootsRank']
    VertexC = G.graph['VertexC']
    Subtree = G.graph['Subtree']
    gnT = G.graph['gnT']
    fnT = np.arange(N + M)

    n2s = NodeStr(np.arange(N), N)
    if debug:
        warn = print
    else:
        def warn(*args, **kwargs):
            pass

    Barrier, barrierLo, barrierHi = get_barrier(G.graph, barrier_gate)

    # dstC = VertexC[dst]
    blocked_ = blocked = org
    goal_ = dst
    cleatnode = gnT[org]

    # print(f'[{i}] <plan_detour_recursive[{depth}]> {n2s(u, v)} '
    #       f'blocking {n2s(blocked, goal_)}')
    goalC = VertexC[goal_]
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

    not2hook = remove.copy()

    store = []
    # look for detours on the Lo and Hi sides of barrier
    for corner_, loNotHi, sidelabel in ((barrierLo, True, 'Lo'),
                                        (barrierHi, False, 'Hi')):
        warn(f'({depth}|{sidelabel}) BEGIN: {n2s(corner_)}')

        # block for possible future change (does nothing)
        nearest_root = -M + np.argmin(d2roots[corner_])
        if nearest_root != root:
            debug and print(f'corner: {n2s(corner_)} is closest to '
                            f'{n2s(nearest_root)} than to {n2s(root)}')

        # block for finding the best hook
        cornerC = VertexC[corner_]
        Blocked = list(set(Subtree[cleatnode]) - remove)

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
        warn(f'prevL: {prevL:.0f}')

        # check if the bend at corner is necessary
        discrim = angle(hookC, cornerC, goalC) > 0
        dropcorner = discrim != loNotHi
        # if hook < N and dropcorner:
        # if dropcorner and False:  # TODO: conclude this test
        if dropcorner and fnT[hook] != corner_:
            warn(f'DROPCORNER {sidelabel}')
            # <bend unnecessary>
            detourL = np.hypot(*(goalC - hookC))
            addedL = prevL - detourL
            # debug and print(f'[{i}] CONCAVE: '
            # print(f'[{i}] CONCAVE: '
            #       f'{n2s(hook, corner_, goal_)}')
            detourX = get_crossings(G, goal_, hook, detour_waiver=True)
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
        BarrierPrime = np.array([b for b in Barrier if b < N])
        BarrierC = VertexC[fnT[BarrierPrime]]

        is_barrier_split, insideI, outsideI = is_bunch_split_by_corner(
            BarrierC, hookC, cornerC, goalC)

        # TODO: think if this gate edge waiver is correct
        FarX = [farX for farX in get_crossings(G, hook, corner_,
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
        nearX = get_crossings(G, corner_, goal_, detour_waiver=True)
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
                # new_barrierLo = ComponLoLim[subbarrier, root]
                # new_barrierHi = ComponHiLim[subbarrier, root]
                remaining_savings = savings - addedL
                # def detour(G, root, org, dst, u, v, barrier_gate, depth=0,
                #            remove=set(), debug=False, maxDepth=5, savings=np.inf):
                subdetour = detour(
                    G, root, hook, corner_, *farX, subbarrier,
                    depth + 1, savings=remaining_savings,
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
                    dcX = get_crossings(G, subcorner_, corner_,
                                        detour_waiver=True)
                    if not dcX:
                        print(f'[{depth}] dropped corner '
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
