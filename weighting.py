# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import numpy as np


class Weight():

    @classmethod
    def blockage_xtra(cls, data):
        arc = data['arc'][data['root']]
        penalty = np.pi/(np.pi - arc) + 4*arc/np.pi
        return data['length']*penalty

    @classmethod
    def blockage(cls, data):
        arc = data['arc'][data['root']]
        penalty = np.pi/(np.pi - arc)
        return data['length']*penalty
