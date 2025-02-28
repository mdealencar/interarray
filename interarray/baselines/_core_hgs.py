# [[2012.10384] Hybrid Genetic Search for the CVRP: Open-Source Implementation
#  and SWAP\* Neighborhood]
# (https://arxiv.org/abs/2012.10384)
from dataclasses import asdict
import hygese as hgs
import numpy as np
from py.io import StdCaptureFD


def _solution_time(log, objective) -> float:
    sol_repr = f'{objective:.2f}'
    for line in log.splitlines():
        if line[0] == '-':
            continue
        fields = line.split(' | ')
        if fields[2] != 'NO-FEASIBLE':
            incumbent = fields[2].split(' ')[2]
        else:
            incumbent = ''
        if incumbent == sol_repr:
            _, time = fields[1].split(' ')
            return float(time)
    # if sol_repr was not found, return total runtime
    return float(line.split(' ')[-1])


def do_hgs(W, coordinates, vehicles, capacity, solver_options):

    demands = np.ones(coordinates.shape[1], dtype=float)
    demands[0] = 0.  # depot demand = 0

    data = dict(
        x_coordinates=coordinates[0],
        y_coordinates=coordinates[1],
        distance_matrix=W,
        demands=demands,
        vehicle_capacity=float(capacity),
        num_vehicles=vehicles,
        depot=0,
    )

    ap = hgs.AlgorithmParameters(**solver_options)
    hgs_solver = hgs.Solver(parameters=ap, verbose=True)

    result, log, _ = StdCaptureFD.call(hgs_solver.solve_cvrp, data,
                                       rounding=False)
    
    solution_time = _solution_time(log, result.cost)

    return (result.routes, result.time, solution_time, result.cost, log,
            asdict(ap))
