# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import os
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple
from importlib.resources import files

import networkx as nx
import numpy as np
import utm
import yaml

from .geometric import make_graph_metrics
from .utils import NodeTagger

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


def graph_from_yaml(filepath, handle=None) -> nx.Graph:
    '''Import wind farm data from .yaml file.'''
    fpath = filepath.with_suffix('.yaml')
    # read wind power plant site YAML file
    parsed_dict = yaml.safe_load(open(fpath, 'r', encoding='utf8'))
    Boundary, BoundaryTag = _tags_and_array_from_key('EXTENTS', parsed_dict)
    Root, RootTag = _tags_and_array_from_key('SUBSTATIONS', parsed_dict)
    Node, NodeTag = _tags_and_array_from_key('TURBINES', parsed_dict)

    # create networkx graph
    N = Node.shape[0]
    M = Root.shape[0]
    G = nx.Graph(M=M,
                 VertexC=np.vstack([Node, Root]),
                 boundary=Boundary,
                 name=fpath.stem,
                 handle=handle)
    lsangle = parsed_dict.get('LANDSCAPE_ANGLE')
    if lsangle is not None:
        G.graph['landscape_angle'] = lsangle
    G.add_nodes_from(((n, {'label': F[n], 'type': 'wtg', 'tag': NodeTag[n]})
                      for n in range(N)))
    G.add_nodes_from(((r, {'label': F[r], 'type': 'oss', 'tag': RootTag[r]})
                      for r in range(-M, 0)))
    make_graph_metrics(G)
    return G


_site_handles = dict(
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


def load_repository(handles2name=_site_handles) -> NamedTuple:
    base_dir = files('interarray.data')
    return namedtuple('SiteRepository', handles2name)(
        *(graph_from_yaml(base_dir / fname, handle)
          for handle, fname in handles2name.items()))
