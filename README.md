interarray
==========

Tool for designing and optimizing the electrical cable network (collection system) for offshore wind power plants.

About interarray
----------------

``interarray`` provides a framework to obtain optimal or near-optimal cable routes for a given turbine layout within the cable-laying boundaries. It provides high-level access to heuristic, meta-heuristic and mathematical optimization approaches to the problem.

The design of the collection system is subject to constraints:
- circuits can only branch inside a wind turbine, if at all;
- cables cannot cross each other;
- cable routes must fall inside the allowed area, avoiding obstacles within it;
- the maximum current capacity of the cable must be respected.

This problem has similarities with two classic operations research problems:
- The capacitated minimum spanning tree problem (CMSTP);
- The open and capacitated vehicle routing problem (OCVRP);

Neither of the classic formulations consider route crossings, which is the main achievement of ``interarray``. Whether the approach is via the CMSTP or via the OCVRP depends on the viability of branching the circuits on turbines.

The heuristics are based on extensions to the Esau-Williams heuristic (for the CMSTP). The meta-heuristic is [implemented elsewhere](https://github.com/vidalt/HGS-CVRP), of which `interarray` is just a user. The mathematical optimization uses mixed-integer linear programming (MILP) models, which can be solved using Google's OR-Tools or by calling solvers via ``pyomo``, e.g.: Coin-OR Branch-and-Cut (CBC), IBM's CPLEX, Gurobi, HiGHS, SCIP, among others.


Requirements
------------

Essential external requirements are in:
- [requirements.txt](requirements.txt) if using `pip`
- [environment.yml](environment.yml) if using `conda`

[PythonCDT](https://github.com/artem-ogre/PythonCDT) is an **essential requirement** that is not `pip`- nor `conda`-installable. Please refer to its repository for installation instructions.

See the *Solvers* section for the **optional requirements** for performing mathematical optimization.

### Setup environment with pip

With you python environment active, call:

```
pip install -r requirements.txt
```

### Setup environment with conda

```
conda env create --name «env_interarray» --file environment.yml
conda activate «env_interarray»
```


Installation
------------

``interarray`` is not yet a proper python package. For the time being:

```
git clone https://github.com/mdealencar/interarray.git
```

And add the `interarray` folder to somewhere Python looks for packages.


Solvers
-------

The installation procedure above enables using the heuristics and the meta-heuristic within ``interarray``. To benefit from mathematical optimization, at least one MILP solver is necessary. Activate your python environment and choose either the `pip` or the `conda` command.

See the MILP [notebooks](notebooks) for relevant parameters when calling each solver.

Only `gurobi`, `cplex` and `cbc` are currently (Jan/2025) able to search the branch-and-bound tree using concurrent threads. This usually accelerates the exploration of the search space in multi-core computers. The `ortools` solver also benefits from multi-core systems by launching a portfolio of algorithms in parallel, with some information exchange among them.

### OR-tools

[Google's OR-Tools](https://developers.google.com/optimization) is open source software.

```
pip install ortools
```

### Gurobi

[Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) is proprietary software (academic license available). The trial version can only handle very small problems.

```
pip install gurobipy
conda install -c gurobi gurobi
```

### CPLEX

[IBM ILOG CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) is proprietary software (academic license available). The Community Edition version can only handle very small problems.

```
pip install cplex
```

### HiGHS

[HiGHS](https://highs.dev/) is open source software.

```
pip install highspy
conda install highspy
```

### SCIP

[SCIP](https://www.scipopt.org/) is open source software.

```
conda install scip
```

### CBC

[COIN-OR's Optimization Suite](https://coin-or.github.io/user_introduction.html) is open source software and its MILP solver is [coin-or/Cbc: COIN-OR Branch-and-Cut solver](https://github.com/coin-or/Cbc).

Pyomo's interface with CBC is through a system call, so it does not need to be part of a python environment, but Pyomo must be able to find the solver's executable file. Conda has a package for it, but it may also be installed by following the instructions in the links above.

```
conda install coin-or-cbc
```


Documentation
-------------

Some usage examples can be found in [notebooks](notebooks).

The heuristics implemented in this repository (release 0.0.1) are presented and analyzed in the MSc thesis [Optimization heuristics for offshore wind power plant collection systems design](https://fulltext-gateway.cvt.dk/oafilestore?oid=62dddf809a5e7116caf943f3&targetid=62dddf80a41ba354e4ed35bc) (DTU Wind - Technical University of Denmark, July 4, 2022)

The meta-heuristic used is [vidalt/HGS-CVRP: Modern implementation of the hybrid genetic search (HGS) algorithm specialized to the capacitated vehicle routing problem (CVRP). This code also includes an additional neighborhood called SWAP\*.](https://github.com/vidalt/HGS-CVRP) via its Python bindings [chkwon/PyHygese: A Python wrapper for the Hybrid Genetic Search algorithm for Capacitated Vehicle Routing Problems (HGS-CVRP)](https://github.com/chkwon/PyHygese).

The cable routing relies on a navigation mesh generated by the library [artem-ogre/CDT: Constrained Delaunay Triangulation (C++)](https://github.com/artem-ogre/CDT) via its Python bindings - [artem-ogre/PythonCDT: Constrained Delaunay Triangulation (Python)](https://github.com/artem-ogre/PythonCDT).
