# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import numpy as np


def translate2global_optimizer(G):
    VertexC = G.graph['VertexC']
    M = G.graph['M']
    X, Y = np.hstack((VertexC[-1:-1 - M:-1].T, VertexC[:-M].T))
    return dict(WTc=G.number_of_nodes() - M, OSSc=M, X=X, Y=Y)
