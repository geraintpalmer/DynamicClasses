"""
Usage:
    write_row_simulation.py <id> <l1> <l2> <m1> <m2> <t1> <t2> <c> <max_sim_time> <warmup> <cooldown>

Arguments
    id           : ID number of the parameter set
    l1           : arrival rate of class 1
    l2           : arrival rate of class 2
    m1           : service rate of class 1
    m2           : service rate of class 2
    t1           : transition rate from class 1 to class 2
    t2           : transition rate from class 2 to class 1
    c            : number of servers
    max_sim_time : the maximum time to run the simulation
    warmup       : the simulation warmup time
    cooldown     : the simulation cooldown time
"""

import sys

sys.path.append("../..")
import models
from csv import writer
import pandas as pd

if __name__ == "__main__":
    args = sys.argv
    id_num = args[1]
    l1 = float(args[2])
    l2 = float(args[3])
    m1 = float(args[4])
    m2 = float(args[5])
    t1 = float(args[6])
    t2 = float(args[7])
    c = int(args[8])
    max_sim_time = float(args[9])
    warmup = float(args[10])
    cooldown = float(args[11])

    row = models.write_row_simulation(
        num_classes=2,
        arrival_rates=[l1, l2],
        service_rates=[m1, m2],
        num_servers=c,
        thetas=[[None, t1], [t2, None]],
        max_simulation_time=max_sim_time,
        warmup_time=warmup,
        cooldown_time=cooldown,
        progress_bar=False,
    )

    with open("all_simulations.csv", "a", newline="") as f:
        writer_object = writer(f)
        writer_object.writerow([id_num] + row)
