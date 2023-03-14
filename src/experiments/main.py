from csv import writer
import itertools
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

import models

num_classes = 2
max_simulation_time = 1000
warmup_time = 100
cooldown_time = 100

arrival_1_space = np.linspace(1, 5, 5)
arrival_2_space = np.linspace(1, 5, 5)
service_1_space = np.linspace(1, 5, 5)
service_2_space = np.linspace(1, 5, 5)
num_servers_space = np.linspace(1, 5, 5, dtype=int)
theta_12_space = np.linspace(1, 5, 5)
theta_21_space = np.linspace(1, 5, 5)

for (
    arrival_1,
    arrival_2,
    service_1,
    service_2,
    num_servers,
    theta_12,
    theta_21,
) in itertools.product(
    arrival_1_space,
    arrival_2_space,
    service_1_space,
    service_2_space,
    num_servers_space,
    theta_12_space,
    theta_21_space,
):
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

    with open("test.csv", "a") as f:
        writer_object = writer(f)
        writer_object.writerow(row)
