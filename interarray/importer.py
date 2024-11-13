# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import os
from itertools import chain
from collections import namedtuple, defaultdict
from pathlib import Path
from typing import NamedTuple, Iterable
from importlib.resources import files
from functools import reduce
from operator import attrgetter

import networkx as nx
import numpy as np
import utm
import yaml
import osmium
import shapely as shp
import shapely.wkb as wkblib

from .geometric import make_graph_metrics
from .utils import NodeTagger
from .interarraylib import L_from_site

F = NodeTagger()


def translate_latlonstr(entries):
    '''
    This requires a very specific string format:
    TAG 11°22.333'N 44°55.666'E
    (no spaces within one coordinate).
    '''
    translated = []
    for entry in entries.splitlines():
        tag, lat, lon = entry.split(' ')
        latlon = []
        for ll in (lat, lon):
            val, hemisphere = ll.split("'")
            deg, sec = val.split('°')
            latlon.append((float(deg) + float(sec)/60)
                          * (1 if hemisphere in ('N', 'E') else -1))
        translated.append((tag, *utm.from_latlon(*latlon)))
    return translated


def _tags_and_array_from_key(key, parsed_dict):
    # separate data into columns
    tags, eastings, northings, zone_numbers, zone_letters = \
        zip(*translate_latlonstr(parsed_dict[key]))
    # all coordinates must belong to the same UTM zone
    assert all(num == zone_numbers[0] for num in zone_numbers[1:])
    assert all(letter == zone_letters[0] for letter in zone_letters[1:])
    return np.c_[eastings, northings], tags


def L_from_yaml(filepath: Path, handle: str | None = None) -> nx.Graph:
    '''Import wind farm data from .yaml file.

    Args:
        filepath: Path of `.yaml` file to read.
        handle: Short moniker for the site.

    Returns:
        Unconnected locations graph L.
    '''
    fpath = Path(filepath)
    # read wind power plant site YAML file
    parsed_dict = yaml.safe_load(open(fpath, 'r', encoding='utf8'))
    Border, BorderTag = _tags_and_array_from_key('EXTENTS', parsed_dict)
    Root, RootTag = _tags_and_array_from_key('SUBSTATIONS', parsed_dict)
    Node, NodeTag = _tags_and_array_from_key('TURBINES', parsed_dict)

    # create networkx graph
    T = Node.shape[0]
    R = Root.shape[0]
    B = Border.shape[0]
    G = nx.Graph(T=T, R=R, B=B,
                 VertexC=np.vstack((Node, Border, Root)),
                 border=np.arange(T, T + B),
                 name=fpath.stem,
                 handle=handle)
    lsangle = parsed_dict.get('LANDSCAPE_ANGLE')
    if lsangle is not None:
        G.graph['landscape_angle'] = lsangle
    G.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg', 'tag': NodeTag[n]})
                      for n in range(T)))
    G.add_nodes_from(((r, {'label': F[r], 'kind': 'oss', 'tag': RootTag[r]})
                      for r in range(-R, 0)))
    make_graph_metrics(G)
    return G


wkbfab = osmium.geom.WKBFactory()


class GetAllData(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.elements = defaultdict(lambda: defaultdict(list))

    def node(self, n):
        power = n.tags.get('power') or n.tags.get('construction:power')
        if power in ['generator', 'substation', 'transformer']:
            self.elements[power]['nodes'].append((
                dict(n.tags),
                wkblib.loads(wkbfab.create_point(n), hex=True)
            ))

    def way(self, w):
        power = w.tags.get('power') or w.tags.get('construction:power')
        if power in ['plant', 'substation', 'transformer']:
            self.elements[power]['ways'].append((
                dict(w.tags),
                wkblib.loads(wkbfab.create_linestring(w), hex=True)
            ))

    def relation(self, r):
        self.elements[r.tags.get('power')
                      or r.tags.get('construction:power')
                      or 'other']['rels'].append(
                          (r.replace(tags=dict(r.tags),
                                     members=list(r.members))))


def L_from_pbf(filepath: Path, handle: str | None = None) -> nx.Graph:
    '''Import wind farm data from .osm.pbf file.

    Args:
        filepath: Path of `.osm.pbf` file to read.
        handle: Short moniker for the site.

    Returns:
        Unconnected locations graph L.
    '''
    fpath = Path(filepath)
    assert ['.osm', '.pbf'] == fpath.suffixes[-2:], \
        'Argument `filepath` does not have `.osm.pbf` extension.'
    name = filepath.stem[:-4]
    # read wind power plant site OpenStreetMap's Protobuffer file
    getter = GetAllData()
    getter.apply_file(fpath, locations=True, idx='flex_mem')
    data = getter.elements

    # Generators
    wtg = []
    gen_nodes = data['generator'].get('nodes')
    if gen_nodes is not None:
        wtg = [point for tags, point in gen_nodes]
    else:
        print(f'«{name}» Unable to identify generators.')
    T = len(wtg)

    # Substations
    ossn = ([point for tags, point in data['substation']['nodes']]
            + [point for tags, point in data['transformer']['nodes']])
    ossw = ([ring.centroid for tags, ring in data['substation']['ways']]
            + [ring.centroid for tags, ring in data['transformer']['ways']])
    if ossn and ossw:
        print(f'«{name}» Warning: substations found both as nodes and ways.')
    oss = ossn + ossw
    R = len(oss)

    # Boundary
    plant_ways = data['plant'].get('ways')
    if plant_ways is not None:
        assert len(plant_ways) == 1, \
                'Not Implemented: power plant with multiple areas.'
        (tags, ring), = plant_ways
        plant_name = tags.get('name:en') or tags.get('name')
        boundary = ring
    else:
        plant_name = None
        print(f'«{name}» Error: No site boundary found.')
    B = len(boundary.coords) - 1

    # Relations (nothing done now, possibly will get plant name from this)
    plant_rels = data['plant'].get('rels')
    if plant_rels is not None:
        print(f'«{name}» Relations found: ', plant_rels)

    # Build site data structure
    latlon = np.array(tuple(chain(
        ((n.y, n.x) for n in wtg),
        boundary.coords.__array__()[:-1, ::-1],
        ((n.y, n.x) for n in oss),
        )), dtype=float
    )
    # TODO: find the UTM sector that includes the most coordinates among
    # vertices and boundary (bin them in 6° sectors and count). Then insert
    # a dummy coordinate as the first array row, such that utm.from_latlon()
    # will pick the right zone for all points.
    VertexC = np.c_[utm.from_latlon(*latlon.T)[:2]]
    # create networkx graph
    # TODO: use the wtg names as node labels if available
    #       this means bringing the core of L_from_site here
    L = L_from_site(
            T=T, R=R, B=B,
            VertexC=VertexC,
            border=np.arange(T, T + B),
            name=name,
            handle=handle,
            )
    if plant_name is not None:
        L.graph['OSM_name'] = plant_name
    x, y = boundary.minimum_rotated_rectangle.exterior.coords.xy
    side0 = np.hypot(x[1] - x[0], y[1] - y[0])
    side1 = np.hypot(x[2] - x[1], y[2] - y[1])
    if side0 < side1:
        angle = np.arctan2((x[1] - x[0]), (y[1] - y[0]))
    else:
        angle = np.arctan2((x[2] - x[1]), (y[2] - y[1]))
    if abs(angle) > np.pi/2:
        angle += np.pi if angle < 0 else -np.pi
    L.graph['landscape_angle'] = 180*angle/np.pi
    make_graph_metrics(L)
    return L


_site_handles_yaml = dict(
    anholt='Anholt',
    borkum='Borkum Riffgrund 1',
    borkum2='Borkum Riffgrund 2',
    borssele='Borssele',
    butendiek='Butendiek',
    dantysk='DanTysk',
    doggerA='Dogger Bank A',
    dudgeon='Dudgeon',
    anglia='East Anglia ONE',
    gode='Gode Wind 1',
    gabbin='Greater Gabbard Inner',
    gwynt='Gwynt y Mor',
    hornsea='Hornsea One',
    hornsea2w='Hornsea Two West',
    horns='Horns Rev 1',
    horns2='Horns Rev 2',
    horns3='Horns Rev 3',
    london='London Array',
    moray='Moray East',
    moraywest='Moray West',
    ormonde='Ormonde',
    race='Race Bank',
    rampion='Rampion',
    rødsand2='Rødsand 2',
    thanet='Thanet',
    triton='Triton Knoll',
    walney1='Walney 1',
    walney2='Walney 2',
    walneyext='Walney Extension',
    sands='West of Duddon Sands',
)

_site_handles_pbf = dict(
    amalia='Pricess Amalia',
    amrumbank='Amrumbank West',
    arkona='Arkona',
    baltic2='Baltic 2',
    bard='BARD Offshore 1',
    beatrice='Beatrice',
    belwind='Belwind',
    binhainorthH2='SPIC Binhai North H2',
    bodhi='Laoting Bodhi Island',
    eagle='Baltic Eagle',
    fecamp='Fecamp',
    gemini1='Gemini 1',
    gemini2='Gemini 2',
    glotech1='Global Tech 1',
    hohesee='Hohe See',
    jiaxing1='Zhejiang Jiaxing 1',
    kfA='Kriegers Flak A',
    kfB='Kriegers Flak B',
    lillgrund='Lillgrund',
    lincs='Lincs',
    meerwind='Meerwind',
    merkur='Merkur',
    nanpeng='CECEP Yangjiang Nanpeng Island',
    neart='Neart na Gaoithe',
    nordsee='Nordsee One',
    northwind='Northwind',
    nysted='Nysted',
    robin='Robin Rigg',
    rudongdemo='CGN Rudong Demonstration',
    rudongH10='Rudong H10',
    rudongH6='Rudong H6',
    rudongH8='Rudong H8',
    sandbank='Sandbank',
    seagreen='Seagreen',
    shengsi2='Shengsi 2',
    sheringham='Sheringham Shoal',
    vejamate='Veja Mate',
    wikinger='Wikinger',
    brieuc='Saint-Brieuc',
    nazaire='Saint-Nazaire',
    riffgat='Riffgat',
    humber='Humber Gateway',
    rough='Westermost Rough',
    bucht='Deutsche Bucht',
    nordseeost='Nordsee Ost',
    kaskasi='Kaskasi',
    albatros='Albatros',
    luchterduinen='Luchterduinen',
    norther='Norther',
    mermaid='Mermaid',
    rentel='Rentel',
    triborkum='Trianel Windpark Borkum',
    galloper='Galloper Inner',
)

_site_handles = _site_handles_yaml | _site_handles_pbf

_READERS = {
        '.yaml': L_from_yaml,
        '.osm.pbf': L_from_pbf,
        }


def load_repository(handles2names=(
        ('.yaml', _site_handles_yaml),
        ('.osm.pbf', _site_handles_pbf),
        )) -> NamedTuple:
    base_dir = files('interarray.data')
    if isinstance(handles2names, dict):
        # assume all files have .yaml extension
        return namedtuple('SiteRepository', handles2name)(
            *(L_from_yaml(base_dir / fname, handle)
              for handle, fname in handles2names.items()))
    elif isinstance(handles2names, Iterable):
        # handle multiple file extensions
        return namedtuple('SiteRepository',
                          reduce(lambda a, b: a | b,
                                 (m for _, m in handles2names)))(
            *sum((tuple(_READERS[ext](base_dir / (fname + ext), handle)
                        for handle, fname in handle2name.items())
                 for ext, handle2name in handles2names),
                 tuple()))
