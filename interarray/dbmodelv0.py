# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import datetime
import os

from pony.orm import Database, IntArray, Optional, PrimaryKey, Required, Set


def open_database(filename, create_db=False):
    db = Database()
    define_entities(db)
    db.bind('sqlite', os.path.abspath(os.path.expanduser(filename)),
            create_db=create_db)
    db.generate_mapping(create_tables=True)
    return db


def define_entities(db):
    '''
    Database model for storage of layouts.
    Tables:
    - NodeSet: site
    - EdgeSet: layout
    - Method: info on algorithm & options to produce layouts
    - Machine: info on machine that generated a layout
    '''

    class NodeSet(db.Entity):
        # hashlib.sha256(VertexC + boundary).digest()
        digest = PrimaryKey(bytes)
        name = Required(str, unique=True)
        handle = Required(str, unique=True)
        T = Required(int)  # # of non-root nodes
        R = Required(int)  # # of root nodes
        # vertices (nodes + roots) coordinates (UTM)
        # pickle.dumps(np.empty((T + R, 2), dtype=float)
        VertexC = Required(bytes)
        # region polygon: P vertices (x, y), ordered ccw
        # pickle.dumps(np.empty((P, 2), dtype=float)
        boundary = Required(bytes)
        landscape_angle = Optional(float)
        EdgeSets = Set(lambda: EdgeSet)

    class EdgeSet(db.Entity):
        nodes = Required(NodeSet)
        # edges = pickle.dumps(
        # np.array([(u, v)
        #           for u, v in G.edges], dtype=int))
        edges = Required(bytes)
        length = Required(float)
        # number of Detour nodes
        D = Optional(int, default=0)
        clone2prime = Optional(IntArray)
        gates = Required(IntArray)
        method = Required(lambda: Method)
        capacity = Required(int)
        # cables = Optional(lambda: CableSet)
        runtime = Optional(float)
        runtime_unit = Optional(str)
        machine = Optional(lambda: Machine)
        timestamp = Optional(datetime.datetime, default=datetime.datetime.utcnow)
        # DetourC = Optional(bytes)  # superceeded by D and clone2prime
        # misc is a pickled python dictionary
        misc = Optional(bytes)

    class Method(db.Entity):
        # hashlib.sha256(funhash + options).digest()
        digest = PrimaryKey(bytes)
        funname = Required(str)
        # hashlib.sha256(esauwilliams.__code__.co_code)
        funhash = Required(bytes)
        # capacity = Required(int)
        # options is a dict of function parameters
        options = Required(str)
        timestamp = Required(datetime.datetime, default=datetime.datetime.utcnow)
        EdgeSets = Set(EdgeSet)

    class Machine(db.Entity):
        name = Required(str, unique=True)
        EdgeSets = Set(EdgeSet)

    # class CableSet(db.Entity):
    #     name = Required(str)
    #     cableset = Required(bytes)
    #     EdgeSets = Set(EdgeSet)
    #     max_capacity = Required(int)
    #     # name = Required(str)
    #     # types = Required(int)
    #     # areas = Required(IntArray)  # mm²
    #     # capacities  = Required(IntArray)  # num of wtg
    #     # EdgeSets = Set(EdgeSet)
    #     # max_capacity = Required(int)
