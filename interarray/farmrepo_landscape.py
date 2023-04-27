# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import numpy as np
from collections import namedtuple
from interarray.fileio import file2graph
from interarray.synthetic import synthfarm2graph, equidistant
from pathlib import Path


def namedtuplify(namedtuple_typename='', **kwargs):
    NamedTuplified = namedtuple(namedtuple_typename,
                                tuple(str(kw) for kw in kwargs))
    return NamedTuplified(**kwargs)


def tess(radius, spacing=1000):
    NodeC = equidistant(radius, center='centroid', spacing=spacing)
    RootC = np.zeros((1, 2), dtype=float)
    return synthfarm2graph(RootC, NodeC, name='SynthTess')


def tess3(radius, spacing=1000):
    h = np.sqrt(3)/2
    NodeC = equidistant(radius, center='vertex', spacing=spacing)
    RootC = 5*spacing*np.array(((3/4, -h/2), (-3/4, -h/2), (0., h)))
    return synthfarm2graph(RootC, NodeC, name='SynthTess (3 OSS)')


def tess3sm(radius, spacing=1000):
    h = np.sqrt(3)/2
    NodeC = equidistant(radius, center='vertex', spacing=spacing)
    RootC = 2.5*spacing*np.array(((-0.5, -h), (-0.5, h), (1., 0.)))
    return synthfarm2graph(RootC, NodeC, name='SynthTess small (3 OSS)')

datapath = Path(__file__).resolve().parent.parent / 'data'

g = namedtuplify(
    namedtuple_typename='FarmGraphs',
    # .xlsx files
    # 100 WTG
    thanet=file2graph(datapath / 'Thanet.xlsx', rotate=49),
    # 80 WTG
    dantysk=file2graph(datapath / 'DanTysk.xlsx', rotate=90),
    # 80 WTG
    horns=file2graph(datapath / 'Horns Rev 1.xlsx'),
    # 111 WTG
    anholt=file2graph(datapath / 'Anholt.xlsx', rotate=84),
    # 108 WTG
    sands=file2graph(datapath / 'West of Duddon Sands.xlsx', rotate=55),
    # 30 WTG
    ormonde=file2graph(datapath / 'Ormonde.xlsx', rotate=45),
    # 175 WTG, 2 OSS
    london=file2graph(datapath / 'London Array.xlsx', rotate=-95),
    # 27 WTG
    rbn=file2graph(datapath / 'BIG Ronne Bank North.xlsx', rotate=-4),
    # 53 WTG
    rbs=file2graph(datapath / 'BIG Ronne Bank South.xlsx', rotate=-2),

    # synthetic farms
    # 114 WTG
    tess=tess(radius=5600),
    # 241 WTG, 3 OSS
    tess3=tess3(radius=8000),
    # ? WTG, 3 OSS
    # tess3sm=tess3sm(radius=5300),
)
