# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

from collections import defaultdict

import numpy as np

import svg

from .geometric import rotate
from .interarraylib import calcload


class SvgRepr():
    '''
    Helper class to get IPython to display the SVG figure encoded in data.
    '''

    def __init__(self, data: str):
        self.data = data

    def _repr_svg_(self) -> str:
        return self.data

    def save(self, filepath: str) -> None:
        '''write SVG to file `filepath`'''
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(self.data)


def svgplot(G, landscape=True, dark=True, node_size=12):
    '''NetworkX graph conversion to SVG'''
    w, h = 1920, 1080
    margin = 30
    root_side = round(1.77*node_size)
    # TODO: ¿use SVG's attr overflow="visible" instead of margin?
    VertexC = G.graph['VertexC']
    BoundaryC = G.graph.get('boundary')
    if BoundaryC is None:
        hull = G.graph.get('hull')
        if hull is not None:
            BoundaryC = VertexC[hull]
        else:
            import shapely as shp
            BoundaryC = np.array(tuple(zip(*shp.MultiPoint(
                G.graph['VertexC']).convex_hull.exterior.coords.xy))[:-1])
    landscape_angle = G.graph.get('landscape_angle')
    if landscape and landscape_angle:
        # landscape_angle is not None and not 0
        VertexC = rotate(VertexC, landscape_angle)
        BoundaryC = rotate(BoundaryC, landscape_angle)

    # viewport scaling
    Woff = min(VertexC[:, 0].min(), BoundaryC[:, 0].min())
    W = max(VertexC[:, 0].max(), BoundaryC[:, 0].max()) - Woff
    Hoff = min(VertexC[:, 1].min(), BoundaryC[:, 1].min())
    H = max(VertexC[:, 1].max(), BoundaryC[:, 1].max()) - Hoff
    wr = (w - 2*margin)/W
    hr = (h - 2*margin)/H
    if wr/hr < w/h:
        r = wr
        h = round(H*r + 2*margin)
    else:
        r = hr
        #  w = round(W*r + 2*margin)
    offset = np.array((Woff, Hoff))
    VertexS = (VertexC - offset)*r + margin
    BoundaryS = (BoundaryC - offset)*r + margin
    # y axis flipping
    VertexS[:, 1] = h - VertexS[:, 1]
    BoundaryS[:, 1] = h - BoundaryS[:, 1]
    VertexS = VertexS.round().astype(int)
    BoundaryS = BoundaryS.round().astype(int)

    # color settings
    type2color = {}
    type2style = dict(
        detour='dashed',
        scaffold='dotted',
        extended='dashed',
        delaunay='solid',
        unspecified='solid',
    )
    if dark:
        type2color.update(
            detour='darkorange',
            scaffold='gray',
            delaunay='darkcyan',
            extended='darkcyan',
            unspecified='crimson',
        )
        root_color = 'lawngreen'
        node_edge = 'none'
        detour_ring = 'orange'
        polygon_edge = '#333'
        polygon_face = '#080808'
    else:
        type2color.update(
            detour='royalblue',
            scaffold='gray',
            delaunay='black',
            extended='black',
            unspecified='firebrick',
        )
        root_color = 'black'
        node_edge = 'black'
        detour_ring = 'deepskyblue'
        polygon_edge = '#444444'
        polygon_face = 'whitesmoke'
    # matplotlib tab20
    colors = ('#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
              '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
              '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
              '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5')

    M = G.graph['M']
    N = VertexC.shape[0] - M
    D = G.graph.get('D', 0)
    if D:
        fnT = G.graph['fnT']
    else:
        fnT = []

    # farm boundary shape
    boundary = svg.Polygon(
        id='boundary',
        stroke=polygon_edge,
        fill=polygon_face,
        points=' '.join(str(c) for c in BoundaryS.flat)
    )

    if not G.graph.get('has_loads', False) and G.number_of_edges() == N + D:
        calcload(G)

    # wtg nodes
    subtrees = defaultdict(list)
    for n, sub in G.nodes(data='subtree', default=19):
        if 0 <= n < N:
            subtrees[sub].append(n)
    svgnodes = []
    for sub, nodes in subtrees.items():
        svgnodes.append(svg.G(
            fill=colors[sub % len(colors)],
            elements=[svg.Use(href='#wtg', x=VertexS[n, 0], y=VertexS[n, 1])
                      for n in nodes]))
    svgnodes = svg.G(id='WTGgrp', elements=svgnodes)

    # oss nodes
    svgroots = svg.G(
        id='OSSgrp',
        elements=[svg.Use(href='#oss', x=VertexS[r, 0] - root_side/2,
                          y=VertexS[r, 1] - root_side/2)
                  for r in range(-M, 0)])
    # Detour nodes
    svgdetours = svg.G(
        id='DTgrp', elements=[svg.Use(href='#dt', x=VertexS[d, 0],
                                      y=VertexS[d, 1]) for d in fnT[N: N + D]])

    # Edges
    class_dict = {'delaunay': 'del',
                  'extended': 'ext',
                  'scaffold': 'scf',
                  None: 'std'}
    edges_with_type = G.edges(data='type', default=None)
    edge_lines = defaultdict(list)
    for u, v, edge_type in edges_with_type:
        if edge_type == 'detour':
            continue
        edge_lines[class_dict[edge_type]].append(
                svg.Line(x1=VertexS[u, 0], y1=VertexS[u, 1],
                         x2=VertexS[v, 0], y2=VertexS[v, 1]))
    edges = [svg.G(id='edges', class_=class_, elements=lines)
             for class_, lines in edge_lines.items()]
    #  for class_, lines in edge_lines.items():
    #      edges.append(svg.G(id='edges', class_=class_, elements=lines))
    # Detour edges as polylines (to align the dashes among overlapping lines)
    Points = []
    for r in range(-M, 0):
        detoured = [n for n in G.neighbors(r) if n >= N]
        for t in detoured:
            s = r
            detour_hops = [s, fnT[t]]
            while True:
                nbr = set(G.neighbors(t))
                nbr.remove(s)
                u = nbr.pop()
                detour_hops.append(fnT[u])
                if u < N:
                    break
                s, t = t, u
            Points.append(' '.join(str(c) for c in VertexS[detour_hops].flat))
    if Points:
        edgesdt = svg.G(
            id='detours', class_='dt',
            elements=[svg.Polyline(points=points) for points in Points])
    else:
        edgesdt = []

    # Defs (i.e. reusable elements)
    def_elements = []
    wtg = svg.Circle(id='wtg',
                     stroke=node_edge, stroke_width=2, r=node_size)
    def_elements.append(wtg)
    oss = svg.Rect(id='oss', fill=root_color, stroke=node_edge, stroke_width=2,
                   width=root_side, height=root_side)
    def_elements.append(oss)
    detour = svg.Circle(id='dt', fill='none', stroke_opacity=0.3,
                        stroke=detour_ring, stroke_width=4, r=23)
    def_elements.append(detour)
    defs = svg.Defs(elements=def_elements)

    # Style
    # TODO: use type2style below
    style = svg.Style(text=(
        f'polyline {{stroke-width: 4}} '
        f'line {{stroke-width: 4}} '
        f'.std {{stroke: {type2color["unspecified"]}}} '
        f'.del {{stroke: {type2color["delaunay"]}}} '
        f'.ext {{stroke: {type2color["extended"]}; stroke-dasharray: 18 15}} '
        f'.scf {{stroke: {type2color["scaffold"]}; stroke-dasharray: 10 10}} '
        f'.dt {{stroke-dasharray: 18 15; fill: none; '
        f'stroke: {type2color["detour"]}}}'))

    # Aggregate all elements in the SVG figure.
    out = svg.SVG(
        viewBox=svg.ViewBoxSpec(0, 0, w, h),
        elements=[style, defs,
                  svg.G(id=G.graph['handle'],
                        elements=[boundary, *edges, edgesdt, svgnodes,
                                  svgroots, svgdetours])])
    return SvgRepr(out.as_str())
