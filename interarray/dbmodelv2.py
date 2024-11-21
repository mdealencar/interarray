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
    - RouteSet: routeset (i.e. a record of G)
    - Method: info on algorithm & options to produce layouts
    - Machine: info on machine that generated a layout
    '''

    class NodeSet(db.Entity):
        # hashlib.sha256(VertexC + boundary).digest()
        name = Required(str, unique=True)
        T = Required(int)  # # of non-root nodes
        R = Required(int)  # # of root nodes
        B = Required(int)  # num_border_vertices
        # vertices (nodes + roots) coordinates (UTM)
        # pickle.dumps(np.empty((R + T + B, 2), dtype=float)
        VertexC = Required(bytes)
        # the first group is the border (ccw), then obstacles (cw)
        # B is sum(constraint_groups)
        constraint_groups = Required(IntArray)
        # indices to VertexC, concatenation of the groups' ordered vertices
        constraint_vertices = Required(IntArray)
        landscape_angle = Optional(float)
        digest = PrimaryKey(bytes)
        RouteSets = Set(lambda: RouteSet)

    class RouteSet(db.Entity):
        id = PrimaryKey(int, auto=True)
        handle = Required(str)
        valid = Optional(bool)
        T = Required(int)  # num_nodes
        R = Required(int)  # num_roots
        capacity = Required(int)
        length = Required(float)
        is_normalized = Required(bool)
        # runtime always in [s]
        runtime = Optional(float)
        num_gates = Required(IntArray)
        # number of contour nodes
        C = Optional(int, default=0)
        # number of detour nodes
        D = Optional(int, default=0)
        # short identifier of routeset origin (redundant with Method)
        creator = Optional(str)
        # relative increase from undetoured routeset to the detoured one
        # detoured_length = (1 + detextra)*undetoured_length
        detextra = Optional(float)
        num_diagonals = Optional(int)
        tentative = Optional(IntArray)
        rogue = Optional(IntArray)
        timestamp = Optional(datetime.datetime,
                             default=datetime.datetime.utcnow)
        misc = Optional(Json)
        stuntC = Optional(bytes)  # coords of border stunts
        # len(clone2prime) == C + D
        clone2prime = Optional(IntArray)
        edges = Required(IntArray)
        nodes = Required(NodeSet)
        method = Required(lambda: Method)
        machine = Optional(lambda: Machine)

    class Method(db.Entity):
        solver_name = Required(str)
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
        RouteSets = Set(RouteSet)

    class Machine(db.Entity):
        name = Required(str, unique=True)
        attrs = Optional(Json)
        RouteSets = Set(RouteSet)

    # class CableSet(db.Entity):
    #     name = Required(str)
    #     cableset = Required(bytes)
    #     RouteSets = Set(RouteSet)
    #     max_capacity = Required(int)
    #     # name = Required(str)
    #     # types = Required(int)
    #     # areas = Required(IntArray)  # mm²
    #     # capacities  = Required(IntArray)  # num of wtg
    #     # RouteSets = Set(RouteSet)
    #     # max_capacity = Required(int)
