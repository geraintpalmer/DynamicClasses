"""
This file is used to generate the commands to run the experiments for the paper.
A parameter sweep is performed over a particular set of parameters. One of the
parameters `arrival_1` is fixed and the other parameters are scaled by a factor.
The scaling factors are given by the lists `arrival_2_proportions`,
`service_1_proportions`, `service_2_proportions`, `theta_12_proportions`,
`theta_21_proportions`. The number of servers is varied over the list
`num_servers_space`.

All these are then written to a file `commands.txt`. This file can be executed
using the command `parallel --jobs 60 < commands.txt` to run the experiments in
60 parallel processes.
"""
from csv import writer
import hashlib
import itertools
import sys

sys.path.append("../..")

num_classes = 2
max_simulation_time = 20000
warmup_time = 2000
cooldown_time = 2000

arrival_1 = 1
arrival_2_proportions = [1 / 2, 2 / 3, 1, 1 / 2, 2]
service_1_proportions = [1 / 2, 1, 2, 3, 4]
service_2_proportions = [1 / 2, 1, 2, 3, 4]
num_servers_space = [1, 2, 3]
theta_12_proportions = [1 / 2, 2 / 3, 1, 1 / 2, 2]
theta_21_proportions = [1 / 2, 2 / 3, 1, 1 / 2, 2]

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

    param_string = f"{arrival_1} {arrival_2} {service_1} {service_2} {theta_12} {theta_21} {num_servers} {max_simulation_time} {warmup_time} {cooldown_time}"

    hash_object = hashlib.md5(param_string.encode("utf-8"))
    pram_set_id = hash_object.hexdigest()

    command = f"python write_row_simulation.py {pram_set_id} {param_string}"
    with open("commands.txt", "a") as f:
        f.write(command + "\n")
