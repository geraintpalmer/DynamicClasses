"""
This file is used to run the experiments for the paper. A parameter sweep is
performed over a particular set of parameters. One of the parameters `arrival_1`
is fixed and the other parameters are scaled by a factor. The scaling factors
are given by the lists `arrival_2_proportions`, `service_1_proportions`,
`service_2_proportions`, `theta_12_proportions`, `theta_21_proportions`. The
number of servers is varied over the list `num_servers_space`. The results are
written to the file `data.csv`.
"""

from csv import writer
import itertools
import sys

sys.path.append("..")

import numpy as np
import pandas as pd

import models

num_classes = 2
max_simulation_time = 20000
warmup_time = 2000
cooldown_time = 200

arrival_1 = 1
arrival_2_proportions = [1 / 3, 1 / 2, 1, 2, 3]
service_1_proportions = [1 / 3, 1 / 2, 1, 2, 3]
service_2_proportions = [1 / 3, 1 / 2, 1, 2, 3]
num_servers_space = [1, 2, 3]
theta_12_proportions = [1 / 3, 1 / 2, 1, 2, 3]
theta_21_proportions = [1 / 3, 1 / 2, 1, 2, 3]

for (
    arrival_2_ratio,
    service_1_ratio,
    service_2_ratio,
    num_servers,
    theta_12_ratio,
    theta_21_ratio,
) in itertools.product(
    arrival_2_proportions,
    service_1_proportions,
    service_2_proportions,
    num_servers_space,
    theta_12_proportions,
    theta_21_proportions,
):
    arrival_2 = arrival_1 * arrival_2_ratio
    service_1 = arrival_1 * service_1_ratio
    service_2 = arrival_1 * service_2_ratio
    theta_12 = arrival_1 * theta_12_ratio
    theta_21 = arrival_1 * theta_21_ratio

    arrival_rates = [float(arrival_1), float(arrival_2)]
    service_rates = [float(service_1), float(service_2)]
    thetas = [[None, float(theta_12)], [float(theta_21), None]]

    row = models.write_row_simulation(
        num_classes=num_classes,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        num_servers=num_servers,
        thetas=thetas,
        max_simulation_time=max_simulation_time,
        warmup_time=warmup_time,
        cooldown_time=cooldown_time,
        progress_bar=True,
    )

    with open("data.csv", "a") as f:
        writer_object = writer(f)
        writer_object.writerow(row)
