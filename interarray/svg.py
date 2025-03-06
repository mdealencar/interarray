# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

from collections import defaultdict
from itertools import chain

import numpy as np
import darkdetect

from ground.base import get_context
import svg

from .geometric import rotate
from .interarraylib import describe_G


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


def svgplot(G, landscape=True, dark=None, infobox: bool = True,
            node_size: int = 12, github_bugfix: bool = True):
    '''Make a NetworkX graph representation directly in SVG.

    Because matplotlib's svg backend does not make efficient use of SVG
    primitives.
    '''
    if dark is None:
        dark = darkdetect.isDark()
    w, h = 1920, 1080
    margin = 30
    root_side = round(1.77*node_size)
    # TODO: ¿use SVG's attr overflow="visible" instead of margin?
    R, T, B = (G.graph[k] for k in 'RTB')
    VertexC = G.graph['VertexC']
    C, D = (G.graph.get(k, 0) for k in 'CD')
    border, obstacles, landscape_angle = (
        G.graph.get(k) for k in 'border obstacles landscape_angle'.split())
    if landscape and landscape_angle:
        # landscape_angle is not None and not 0
        VertexC = rotate(VertexC, landscape_angle)
    if border is None:
        hull = G.graph.get('hull')
        if hull is not None:
            border = VertexC[hull]
        else:
            context = get_context()
            Point = context.point_cls
            PointMap = {Point(float(x), float(y)): i for i, (x, y) in enumerate(VertexC)}
            BorderPt = context.points_convex_hull(PointMap.keys())
            border = np.array([PointMap[point] for point in BorderPt])

    # viewport scaling
    idx_B = T + B
    Woff = min(VertexC[:idx_B, 0].min(), VertexC[-R:, 0].min())
    W = max(VertexC[:idx_B, 0].max(), VertexC[-R:, 0].max()) - Woff
    Hoff = min(VertexC[:idx_B, 1].min(), VertexC[-R:, 1].min())
    H = max(VertexC[:idx_B, 1].max(), VertexC[-R:, 1].max()) - Hoff
    wr = (w - 2*margin)/W
    hr = (h - 2*margin)/H
    if W/H < w/h:
        # tall aspect
        r = hr
    else:
        # wide aspect
        r = wr
        h = round(H*r + 2*margin)
    offset = np.array((Woff, Hoff))
    VertexS = (VertexC - offset)*r + margin
    # y axis flipping
    VertexS[:, 1] = h - VertexS[:, 1]
    VertexS = VertexS.round().astype(int)

    # theme settings
    kind2alpha = defaultdict(lambda: 1.)
    kind2alpha['virtual'] = 0.4
    kind2color = {}
    kind2dasharray = dict(
            tentative='18 15',
            rogue='25 5',
            extended='18 15',
            contour_extended='18 15',
            scaffold='10 10',
    )
    if dark:
        kind2color.update(
            scaffold='gray',
            delaunay='darkcyan',
            extended='darkcyan',
            tentative='red',
            rogue='yellow',
            contour_delaunay='green',
            contour_extended='green',
            contour='red',
            planar='darkorchid',
            constraint='purple',
            border = 'silver',
            unspecified='crimson',
            detour='darkorange',
            virtual='gold',
        )
        text_color = 'white'
        root_color = 'lawngreen'
        node_edge = 'none'
        detour_ring = 'orange'
        border_face = '#111'
    else:
        kind2color.update(
            scaffold='gray',
            delaunay='darkgreen',
            extended='darkgreen',
            tentative='darkorange',
            rogue='magenta',
            contour_delaunay='firebrick',
            contour_extended='firebrick',
            contour='black',
            planar='darkorchid',
            constraint='darkcyan',
            border = 'dimgray',
            unspecified='firebrick',
            detour='royalblue',
            virtual='gold',
        )
        text_color = 'black'
        root_color = 'black'
        node_edge = 'black'
        detour_ring = 'deepskyblue'
        border_face = '#eee'
    # matplotlib tab20
    colors = ('#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
              '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
              '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
              '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5')

    fnT = G.graph.get('fnT')
    if fnT is None:
        fnT = np.arange(R + T + B + 3)
        fnT[-R:] = range(-R, 0)

    #############################
    # generate the SVG elements #
    #############################
    styleEntries = []
    # elements should be added according to the desired z-order
    graphElements = []

    # prepare obstacles
    draw_obstacles = []
    if obstacles is not None:
        for obstacle in obstacles:
            draw_obstacles.append(
                'M' + ' '.join(str(c) for c in VertexS[obstacle].flat) + 'z')
    # border with obstacles as holes
    if border is not None:
        borderE = svg.Path(
            id='border',
            stroke=kind2color['border'],
            stroke_dasharray=[15, 7],
            stroke_width=2,
            fill=border_face,
            # fill_rule "evenodd" is agnostic to polygon vertices orientation
            # "nonzero" would depend on orientation (if opposite, no fill)
            fill_rule='evenodd',
            d=' '.join(chain(
                ('M' + ' '.join(str(c) for c in VertexS[border].flat) + 'z',),
                draw_obstacles
            )),
        )
        graphElements.append(borderE)

    # Edges
    class_dict = {'delaunay': 'del',
                  'tentative': 'ttt',
                  'rogue': 'rog',
                  'contour': 'con',
                  'contour_delaunay': 'cod',
                  'contour_extended': 'coe',
                  'extended': 'ext',
                  'scaffold': 'scf',
                  'unspecified': 'std'}
    edges_with_kind = G.edges(data='kind')
    edge_lines = defaultdict(list)
    edge_kinds_used = set()
    for u, v, edge_kind in edges_with_kind:
        if edge_kind == 'detour':
            # detours are drawn separately as polylines
            continue
        if edge_kind is None:
            edge_kind = 'unspecified'
        edge_kinds_used.add(edge_kind)
        u, v = (u, v) if u < v else (v, u)
        edge_lines[class_dict[edge_kind]].append(
            svg.Line(x1=VertexS[fnT[u], 0], y1=VertexS[fnT[u], 1],
                     x2=VertexS[fnT[v], 0], y2=VertexS[fnT[v], 1]))
    if edge_lines:
        styleEntries.append(f'line {{stroke-width: 4}}')
        for edge_kind in edge_kinds_used:
            attrs = dict(stroke=kind2color[edge_kind])
            if edge_kind in kind2dasharray:
                attrs['stroke-dasharray'] = kind2dasharray[edge_kind]
            styleEntries.append(
                f'.{class_dict[edge_kind]} '
                f'{{{"; ".join(f"{k}: {v}" for k, v in attrs.items())}}}'
            )
        edgesE_ = [svg.G(id='edges', class_=[class_], elements=lines)
                   for class_, lines in edge_lines.items()]
        graphElements.extend(edgesE_)

    # detour elements
    if D > 0:
        # Detour edges as polylines (to align the dashes among overlapping lines)
        Points = []
        for r in range(-R, 0):
            detoured = [n for n in G.neighbors(r) if n >= T + B + C]
            for t in detoured:
                s = r
                hops = [s, fnT[t]]
                while True:
                    nbr = set(G.neighbors(t))
                    nbr.remove(s)
                    u = nbr.pop()
                    hops.append(fnT[u])
                    if u < T:
                        break
                    s, t = t, u
                Points.append(' '.join(str(c) for c in VertexS[hops].flat))
        edgesdtE = svg.G(
            id='detours', class_=['dt'],
            elements=[svg.Polyline(points=points) for points in Points])
        graphElements.append(edgesdtE)

        # Detour nodes
        svgdetoursE = svg.G(
            id='DTgrp', elements=[
                svg.Use(href='#dt', x=VertexS[d, 0], y=VertexS[d, 1])
                for d in fnT[T + B + C: T + B + C + D]]
        )
        graphElements.append(svgdetoursE)
        styleEntries.extend((
            f'polyline {{stroke-width: 4}}',
            f'.dt {{stroke-dasharray: 18 15; fill: none; '
                f'stroke: {kind2color["detour"]}}}',
        ))

    # wtg nodes
    subtrees = defaultdict(list)
    for n, sub in G.nodes(data='subtree', default=19):
        if 0 <= n < T:
            subtrees[sub].append(n)
    svgnodes = []
    for sub, nodes in subtrees.items():
        svgnodes.append(svg.G(
            fill=colors[sub % len(colors)],
            elements=[svg.Use(href='#wtg', x=VertexS[n, 0], y=VertexS[n, 1])
                      for n in nodes]))
    svgnodesE = svg.G(id='WTGgrp', elements=svgnodes)
    graphElements.append(svgnodesE)

    # oss nodes
    svgrootsE = svg.G(
        id='OSSgrp',
        elements=[svg.Use(href='#oss', x=VertexS[r, 0] - root_side/2,
                          y=VertexS[r, 1] - root_side/2)
                  for r in range(-R, 0)])
    graphElements.append(svgrootsE)

    # Defs (i.e. reusable elements)
    reusableE = [
        svg.Circle(id='wtg', stroke=node_edge, stroke_width=2, r=node_size),
        svg.Rect(id='oss', fill=root_color, stroke=node_edge, stroke_width=2,
                 width=root_side, height=root_side),
    ]
    if D > 0:
        reusableE.append(svg.Circle(id='dt', fill='none', stroke_opacity=0.3,
                         stroke=detour_ring, stroke_width=4, r=23))

    # Aggregate the SVG root elements
    rootElements = [
        svg.Style(text=' '.join(styleEntries)), svg.Defs(elements=reusableE),
        svg.G(id=G.graph.get('handle', G.graph.get('name', 'handleless')),
              elements=graphElements),
        ]

    # Infobox
    if infobox and G.graph.get('has_loads', False):
        w_drawn = round(W*r + 2*margin)
        desc_lines = describe_G(G)[::-1]

        if github_bugfix:
            # this is a workaround for GitHub's bug in rendering svg utf8 text
            # (only when the svg is inside an ipynb notebook)
            desc_lines = [l.encode('ascii', 'xmlcharrefreplace').decode()
                          for l in desc_lines]

        linesE = [
            svg.TSpan(x=w_drawn, dx=svg.Length(-0.2, 'em'),
                      dy=svg.Length((-1.2 if i else -0.2), 'em'), text=line)
            for i, line in enumerate(desc_lines)
        ]
        rootElements.append(
            svg.Text(x=w_drawn, y=h, elements=linesE, fill=text_color, font_size=40,
                     text_anchor='end', font_family='sans-serif')
        )

    # Aggregate all elements in the SVG figure.
    out = svg.SVG(
        viewBox=svg.ViewBoxSpec(0, 0, w, h),
        elements=rootElements,
    )
    return SvgRepr(out.as_str())
