# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import math
from collections.abc import Sequence
from itertools import chain

import darkdetect
from matplotlib.axes import Axes
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .geometric import rotate
from .interarraylib import NodeTagger


FONTSIZE_LABEL = 5
FONTSIZE_LOAD = 7
FONTSIZE_ROOT_LABEL = 6
FONTSIZE_LEGEND_BOX = 7
FONTSIZE_LEGEND_STRIP = 6
NODESIZE = 18
NODESIZE_LABELED = 70
NODESIZE_LABELED_ROOT = 28
NODESIZE_DETOUR = 80
NODESIZE_LABELED_DETOUR = 150

F = NodeTagger()


def gplot(G: nx.Graph, ax: Axes | None = None,
          node_tag: str | None = None,
          landscape: bool = True, infobox: bool = True,
          scalebar: tuple[float, str] | None = None,
          hide_ST: bool = True, legend: bool = False,
          min_dpi: int = 192, dark=None) -> Axes:
    '''Plot site and routeset contained in G.

    Args:
        ax: Axes instance to plot into. If `None`, opens a new figure.
        node_tag: text label inside each node `None`, 'load' or 'label' (or
            any of the nodes' attributes).
        landscape: True -> rotate the plot by G's attribute 'landscape_angle'.
        infobox: Draw text box with summary of G's main properties: capacity,
            number of turbines, number of feeders, total cable length.
        scalebar: (span_in_data_units, label) add a small bar to indicate the
            plotted features' scale (lower right corner).
        hide_ST: If coordinates include a Delaunay supertriangle, adjust the
            viewport to fit only the actual vertices (i.e. no ST vertices).
        legend: Add description of linestyles and node shapes.
        min_dpi: Minimum dots per inch to use. matplotlib's default is used if
            it is greater than this value.

    Returns:
        Axes instance containing the plot.
    '''
    if dark is None:
        dark = darkdetect.isDark()

    root_size = NODESIZE_LABELED_ROOT if node_tag is not None else NODESIZE
    detour_size = (NODESIZE_LABELED_DETOUR
                   if node_tag is not None else
                   NODESIZE_DETOUR)
    node_size = NODESIZE_LABELED if node_tag is not None else NODESIZE

    kind2color = {}
    kind2style = dict(
        detour=(0, (4, 4)),
        scaffold='dotted',
        extended='dashed',
        delaunay='solid',
        tentative='dashdot',
        rogue='dashed',
        contour_delaunay='solid',
        contour_extended='dashed',
        contour='solid',
        planar='dashdot',
        constraint='solid',
        unspecified='solid',
        border='dashed',
    )
    if dark:
        kind2color.update(
            detour='darkorange',
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
        )
        root_color = 'lawngreen'
        node_edge = 'none'
        detour_ring = 'orange'
        border_face = '#111'
    else:
        kind2color.update(
            detour='royalblue',
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
            unspecified='black',
        )
        #  root_color = 'black' if node_tag is None else 'yellow'
        root_color = 'black'
        node_edge = 'black'
        detour_ring = 'deepskyblue'
        border_face = '#eee'

    R, T, B = (G.graph[k] for k in 'RTB')
    VertexC = G.graph['VertexC']
    C, D = (G.graph.get(k, 0) for k in 'CD')
    border, obstacles, landscape_angle = (
        G.graph.get(k) for k in 'border obstacles landscape_angle'.split())
    if landscape and landscape_angle:
        # landscape_angle is not None and not 0
        VertexC = rotate(VertexC, landscape_angle)

    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw=dict(
                aspect='equal', xmargin=0.002, ymargin=0.002
            ),
            layout='constrained', facecolor='none',
            dpi=max(min_dpi, plt.rcParams['figure.dpi'])
        )
    else:
        ax.set(aspect='equal')
    ax.axis(False)
    # draw farm border
    if border is not None:
        border_opt = dict(facecolor=border_face, linestyle='dashed',
            edgecolor=kind2color['border'], linewidth=0.5)
        borderC = VertexC[border] 
        if obstacles is None:
            ax.fill(*borderC.T, **border_opt)
        else:
            obstacleC_ = [VertexC[obstacle] for obstacle in obstacles]
            # path for the external border
            codes = [Path.MOVETO] + (borderC.shape[0] - 1)*[Path.LINETO] + [Path.CLOSEPOLY]
            points = [row for row in borderC] + [borderC[0]]
            # paths for the obstacle borders
            for obstacleC in obstacleC_:
                codes.extend([Path.MOVETO] + (obstacleC.shape[0] - 1)*[Path.LINETO] + [Path.CLOSEPOLY])
                points.extend([row for row in obstacleC] + [obstacleC[0]])
            # create and add matplotlib artists
            path = Path(points, codes)
            patch = PathPatch(path, **border_opt)
            ax.add_patch(patch)
            ax.autoscale()

    # setup
    roots = range(-R, 0)
    pos = (dict(enumerate(VertexC[:-R]))
           | dict(enumerate(VertexC[-R:], start=-R)))
    if C > 0 or D > 0:
        fnT = G.graph['fnT']
        contour = range(T + B, T + B + C)
        detour = range(T + B + C, T + B + C + D)
        pos |= dict(zip(detour, VertexC[fnT[detour]]))
        pos |= dict(zip(contour, VertexC[fnT[contour]]))
    RootL = {r: G.nodes[r].get('label', F[r]) for r in roots[::-1]}

    colors = plt.get_cmap('tab20', 20).colors
    # default value for subtree (i.e. color for unconnected nodes)
    # is the last color of the tab20 colormap (i.e. 19)
    subtrees = G.nodes(data='subtree', default=19)
    node_colors = [colors[subtrees[n] % len(colors)] for n in range(T)]

    edges_width = 0.7
    edges_capstyle = 'round'
    # draw edges
    base_layer = ('scaffold',)
    for edge_kind in base_layer:
        art = nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=kind2color[edge_kind], label=edge_kind,
            style=kind2style[edge_kind], width=edges_width,
            edgelist=[(u, v) for u, v, kind in G.edges.data('kind')
                      if kind == edge_kind])
        if art:
            art.set_capstyle(edges_capstyle)
    art = nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=kind2color['unspecified'], label='direct',
        style=kind2style['unspecified'], width=edges_width,
        edgelist=[(u, v) for u, v, kind in G.edges.data('kind')
                  if kind is None])
    if art:
        art.set_capstyle(edges_capstyle)
    for edge_kind in kind2style:
        if edge_kind in base_layer:
            continue
        art = nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=kind2color[edge_kind], label=edge_kind,
            style=kind2style[edge_kind], width=edges_width,
            edgelist=[(u, v) for u, v, kind in G.edges.data('kind')
                      if kind == edge_kind])
        if art:
            art.set_capstyle(edges_capstyle)

    # draw nodes
    if D:
        # draw circunferences around nodes that have Detour clones
        arts = nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=detour, alpha=0.4, edgecolors=detour_ring,
            node_color='none', node_size=detour_size, label='corner')
        arts.set_clip_on(False)
    arts = nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=roots, linewidths=0.2, node_color=root_color,
        edgecolors=node_edge, node_size=root_size, node_shape='s', label='OSS')
    arts.set_clip_on(False)
    arts = nx.draw_networkx_nodes(
        G, pos, nodelist=range(T), edgecolors=node_edge, ax=ax, label='WTG',
        node_color=node_colors, node_size=node_size, linewidths=0.2)
    arts.set_clip_on(False)

    # draw labels
    font_size = dict(load=FONTSIZE_LOAD,
                     label=FONTSIZE_LABEL,
                     tag=FONTSIZE_ROOT_LABEL)
    if node_tag is not None:
        if node_tag == 'load' and 'has_loads' not in G.graph:
            node_tag = 'label'
        labels = nx.get_node_attributes(G, node_tag)
        for root in roots:
            if root in labels:
                labels.pop(root)
        if D:
            for det in chain(contour, detour):
                if det in labels:
                    labels.pop(det)
        for n in range(T):
            if n not in labels:
                labels[n] = F[n]
        arts = nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
                                       font_size=font_size[node_tag])
        for artist in arts.values():
            artist.set_clip_on(False)
    # root nodes' labels
    if node_tag is not None:
        arts = nx.draw_networkx_labels(
            G, pos, ax=ax, labels=RootL, font_size=FONTSIZE_ROOT_LABEL,
            font_color='black' if dark else 'yellow')
        for artist in arts.values():
            artist.set_clip_on(False)

    if scalebar is not None:
        bar = AnchoredSizeBar(ax.transData, *scalebar, 'lower right',
                              frameon=False)
        ax.add_artist(bar)

    if infobox:
        capacity = G.graph.get('capacity')
        if capacity is not None:
            info = [f'$\\kappa$ = {capacity}, $T$ = {T}']
            feeder_info = [f'$\\phi_{{{rootL}}}$ = {G.degree[r]}'
                           for r, rootL in RootL.items()]
            min_feeder = math.ceil(T/capacity)
            info.append(f'({sum(G.degree[r] for r in roots) - min_feeder:+d}) '
                        + ', '.join(feeder_info))
            Gʹ = nx.subgraph_view(G,
                                  filter_edge=lambda u, v: 'length' in G[u][v])
            length = Gʹ.size(weight="length")
            if length > 0:
                intdigits = int(np.floor(np.log10(length))) + 1
                info.append(f'Σl = {round(length, max(0, 5 - intdigits))} m')
        if ('has_costs' in G.graph):
            info.append('{:.0f} €'.format(G.size(weight='cost')))
        if 'capacity' in G.graph:
            infobox = ax.legend([], fontsize=FONTSIZE_LEGEND_BOX,
                                title='\n'.join(info), labelspacing=0)
            #                      loc='upper right'
            #                      bbox_to_anchor=(-0.04, 0.80, 1.08, 0)
            #                      bbox_to_anchor=(-0.04, 1.03, 1.08, 0)
            plt.setp(infobox.get_title(), multialignment='center')
    if legend:
        ax.legend(ncol=8, fontsize=FONTSIZE_LEGEND_STRIP, loc='lower center',
                  frameon=False, bbox_to_anchor=(0.5, -0.07),
                  columnspacing=1, handletextpad=0.3)
        if 'capacity' in G.graph and infobox:
            ax.add_artist(infobox)
    if hide_ST and VertexC.shape[0] > R + T + B:
        # coordinates include the supertriangle, adjust view limits to hide it
        nonStC = np.r_[VertexC[:T + B], VertexC[-R:]]
        minima = np.min(nonStC, axis=0)
        maxima = np.max(nonStC, axis=0)
        xmargin, ymargin = abs(maxima - minima)*0.05
        (xlo, xhi), (ylo, yhi) = zip(minima, maxima)
        ax.set_xlim(xlo - xmargin, xhi + xmargin)
        ax.set_ylim(ylo - ymargin, yhi + ymargin)
    return ax


def pplot(P: nx.PlanarEmbedding, A: nx.Graph, **kwargs) -> Axes:
    '''Plot PlanarEmbedding `P` using coordinates from `A`.

    Wrapper for `interarray.plotting.gplot()`. Performs what one would expect
    from `gplot(P, ...)` - which does not work because P lacks coordinates and
    node 'kind' attribute. The source needs to be `A` (as opposed to `G` or
    `L`) because only `A` has the supertriangle's vertices coordinates.

    Args:
        P: Planar embedding to plot.
        A: source of vertex coordinates and 'kind'.

    Returns:
        Axes instance containing the plot.
    '''
    H = nx.create_empty_copy(A)
    if 'has_loads' in H.graph:
        del H.graph['has_loads']
    R, T, B = (A.graph[k] for k in 'RTB')
    H.add_edges_from(P.edges, kind='planar')
    fnT = np.arange(R + T + B + 3)
    fnT[-R:] = range(-R, 0)
    H.graph['fnT'] = fnT
    return gplot(H, **kwargs)


def compare(positional=None, **title2G_dict):
    '''
    Plot layouts side by side. dict keys are inserted in the title.
    Arguments must be either a sequence of graphs or multiple
    `keyword`=«graph_instance»`.
    '''
    if positional is not None:
        if isinstance(positional, Sequence):
            title2G_dict |= {chr(i): val for i, val in
                             enumerate(positional, start=ord('A'))}
        else:
            title2G_dict[''] = positional
    fig, axes = plt.subplots(1, len(title2G_dict), squeeze=False)
    for ax, (title, G) in zip(axes.ravel(), title2G_dict.items()):
        gplot(G, ax=ax, node_tag=None)
        creator = G.graph.get("creator", 'no edges')
        ax.set_title(f'{title} – {G.graph["name"]} '
                     f'({creator})')


def scaffolded(G: nx.Graph, P: nx.PlanarEmbedding) -> nx.Graph:
    scaff = P.to_undirected()
    scaff.graph.update(G.graph)
    for attr in 'fnT C'.split():
        if attr in scaff.graph:
            del scaff.graph[attr]
    R, T, B, C, D = (G.graph.get(k, 0) for k in 'R T B C D'.split())
    nx.set_edge_attributes(scaff, 'scaffold', name='kind')
    constraints = P.graph.get('constraint_edges', [])
    for edge in constraints:
        scaff.edges[edge]['kind'] = 'constraint'
    for n, d in scaff.nodes(data=True):
        if n not in G.nodes:
            continue
        d.update(G.nodes[n])
    if C > 0 or D > 0:
        fnT = G.graph['fnT']
    else:
        fnT = np.arange(R + T + B + C + D)
        fnT[-R:] = range(-R, 0)
    for u, v in G.edges:
        st = fnT[u], fnT[v]
        if st in scaff.edges and 'kind' in scaff.edges[st]:
            del scaff.edges[st]['kind']
    VertexC = G.graph['VertexC']
    supertriangleC = P.graph['supertriangleC']
    if G.graph.get('is_normalized'):
        supertriangleC = G.graph['norm_scale']*(supertriangleC
                                                - G.graph['norm_offset'])
    VertexC = np.vstack((VertexC[:-R],
                         supertriangleC,
                         VertexC[-R:]))
    scaff.graph.update(VertexC=VertexC, fnT=fnT)
    return scaff
