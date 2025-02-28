# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

from heapq import heappop, heappush
import math
import networkx as nx


def clusterize(A: nx.Graph, capacity: int) -> tuple[list[set[int]], list[int]]:
    '''Partition the terminals of A into one cluster per root.

    For the moment, it enforces the minimum number of feeders, i.e.
    ceil(T/capacity).

    The algorithm guarantees that the number of feeders for the entire location
    is not increased by the clustering. This means only one partition may have
    a subtree with capacity slack. It does not attempt to make uniform-sized
    clusters, terminals tend to be allocated to the closest root (distance 
    measured in `P_paths` - see `make_planar_embedding()`).
    '''
    R, T = (A.graph[k] for k in 'RT')
    d2roots = A.graph['d2roots']
    mainheap = []
    idx_ = [tuple(range(-R, i)) + tuple(range(i + 1, 0)) for i in range(-R, 0)]
    expel_ = [[] for _ in range(R)]
    cluster_ = [set() for _ in range(R)]

    # initialize mainheap
    closest_root = -R + d2roots[:T].argmin(axis=1)
    for n, r in enumerate(closest_root):
        d = d2roots[n, r]
        heappush(mainheap, (d, n, r))

    total_feeders = math.ceil(T/capacity)
    num_slack = total_feeders*capacity - T

    # expeller function
    def expel_from(r, blocked=[]):
        expel = expel_[r]
        cluster = cluster_[r]
        saved_ = []
        while expel:
            tradeoff_exp, n_exp, r_exp = heappop(expel)
            if r_exp in blocked:
                # blocked prevents a node being expelled back and forth
                saved_.append((tradeoff_exp, n_exp, r_exp))
                continue
            if n_exp in cluster:
                break
        for saved in saved_:
            heappush(expel, saved)
        was_cluster_exp_round = (len(cluster_[r_exp]) % capacity) == 0
        cluster.remove(n_exp)
        #  print(f'{F[n_exp]}: {F[r]} -> {F[r_exp]}', end='|')
        cluster_[r_exp].add(n_exp)
        d_exp = d2roots[n_exp, r_exp]
        for i in idx_[r_exp]:
            heappush(expel_[r_exp], (d2roots[n_exp, i] - d_exp, n_exp, i))
            # print(f'({d2roots[n_exp, i]:.0f}, {d_exp:.0f})', end='|')
        if was_cluster_exp_round:
            expel_from(r_exp, blocked + [r])

    while mainheap:
        d, n, r = heappop(mainheap)
        cluster = cluster_[r]
        expel = expel_[r]
        was_cluster_round = (len(cluster) % capacity) == 0
        # the (- 1) is because we just popped a node but have not assigned it
        threshold = sum((capacity - len(clu) % capacity)
                        for clu in cluster_) - num_slack - 1
        # add first and expel later if necessary
        cluster.add(n)
        for i in idx_[r]:
            heappush(expel, (d2roots[n, i] - d, n, i))
        if (len(mainheap) <= threshold
                and was_cluster_round
                and not all((len(cluster_[i]) % capacity == 0)
                            for i in idx_[r])):
            # cluster is overfull: expel
            expel_from(r)

    # this only works because the slack is allocated in a single cluster
    num_slack_ = [num_slack if (len(cluster) % capacity) else 0 for cluster in cluster_]
    return cluster_, num_slack_
