# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import datetime
import os

from pony.orm import (Database, IntArray, Json, Optional, PrimaryKey, Required,
                      Set)


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
    - EdgeSet: routeset
    - Method: info on algorithm & options to produce layouts
    - Machine: info on machine that generated a layout
    '''

    class NodeSet(db.Entity):
        # hashlib.sha256(VertexC + boundary).digest()
        name = Required(str, unique=True)
        N = Required(int)  # # of non-root nodes
        M = Required(int)  # # of root nodes
        # vertices (nodes + roots) coordinates (UTM)
        # pickle.dumps(np.empty((M + N + B, 2), dtype=float)
        VertexC = Required(bytes)
        # the first group is the border (ccw), then exclusions (cw)
        # B is sum(constraint_groups)
        constraint_groups = Required(IntArray)
        # indices to VertexC, concatenation of the groups' ordered vertices
        constraint_vertices = Required(IntArray)
        landscape_angle = Optional(float)
        digest = PrimaryKey(bytes)
        EdgeSets = Set(lambda: EdgeSet)

    class EdgeSet(db.Entity):
        id = PrimaryKey(int, auto=True)
        handle = Required(str)
        capacity = Required(int)
        length = Required(float)
        # runtime always in [s]
        runtime = Optional(float)
        machine = Optional(lambda: Machine)
        gates = Required(IntArray)
        N = Required(int)
        M = Required(int)
        # number of contour nodes
        C = Optional(int, default=0)
        # number of detour nodes
        D = Optional(int, default=0)
        diagonals_used = Optional(int)
        timestamp = Optional(datetime.datetime,
                             default=datetime.datetime.utcnow)
        misc = Optional(Json)
        # len(clone2prime) == C + D
        clone2prime = Optional(IntArray)
        edges = Required(IntArray)
        nodes = Required(NodeSet)
        method = Required(lambda: Method)

    class Method(db.Entity):
        funname = Required(str)
        # options is a dict of function parameters
        options = Required(Json)
        timestamp = Required(datetime.datetime,
                             default=datetime.datetime.utcnow)
        funfile = Required(str)
        # hashlib.sha256(fun.__code__.co_code)
        funhash = Required(bytes)
        # hashlib.sha256(funhash + pickle(options)).digest()
        digest = PrimaryKey(bytes)
        EdgeSets = Set(EdgeSet)

    class Machine(db.Entity):
        name = Required(str, unique=True)
        attrs = Optional(Json)
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
