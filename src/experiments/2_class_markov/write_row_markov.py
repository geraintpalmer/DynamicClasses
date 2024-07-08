"""
Usage:
    write_row_markov.py <id> <l1> <l2> <m1> <m2> <t1> <t2> <c> <bound_initial> <bound_final> <bound_step> <reasonable_ratio> <epsilon>

Arguments
    id               : ID number of the parameter set
    l1               : arrival rate of class 1
    l2               : arrival rate of class 2
    m1               : service rate of class 1
    m2               : service rate of class 2
    t1               : transition rate from class 1 to class 2
    t2               : transition rate from class 2 to class 1
    c                : number of servers
    bound_initial    : the initial bound to try
    bound_final      : the final possible bound to try
    bound_step       : the step size to increase the bound
    reasonable_ratio : the reasonable ratio
    epsilon          : the epsilon check
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
    bi = int(args[9])
    bf = int(args[10])
    bs = int(args[11])
    rr = float(args[12])
    e = float(args[13])

    row = models.write_row_markov(
        num_classes=2,
        arrival_rates=[l1, l2],
        service_rates=[m1, m2],
        num_servers=c,
        thetas=[[None, t1], [t2, None]],
        bound_initial=bi,
        bound_final=bf,
        bound_step=bs,
        reasonable_ratio=rr,
        epsilon=e,
    )

    with open("all_markov.csv", "a", newline="") as f:
        writer_object = writer(f)
        writer_object.writerow([id_num] + row)
