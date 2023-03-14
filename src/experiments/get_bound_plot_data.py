import sys
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append("..")
import models
import tqdm
from csv import writer
import multiprocessing
import argparse
import pandas as pd
import random


def get_max_hitting_prob(
    num_classes, num_servers, arrival_rates, service_rates, thetas, bound, rr
):
    (
        state_space,
        trans_matrix,
    ) = models.build_state_space_and_transition_matrix_sojourn_mc(
        num_classes=num_classes,
        num_servers=num_servers,
        arrival_rates=arrival_rates,
        service_rates=service_rates,
        thetas=thetas,
        bound=bound,
    )
    max_hitting_prob = models.get_maximum_hitting_probs(
        state_space=state_space,
        transition_matrix=trans_matrix,
        boundary=bound,
        reasonable_ratio=rr,
    )
    return max_hitting_prob


def write_row(params, bound):
    done = pd.read_csv("bound_hitting_prob_data.csv")
    current = done[
        (done["Example"] == params["example_name"]) & (done["Bound"] == bound)
    ]
    if len(current) == 0:
        print(f"Doing: {params['example_name']} with bound {bound}")
        p = get_max_hitting_prob(
            num_classes=params["num_classes"],
            num_servers=params["num_servers"],
            arrival_rates=params["arrival_rates"],
            service_rates=params["service_rates"],
            thetas=params["thetas"],
            bound=bound,
            rr=0.8,
        )

        with open("bound_hitting_prob_data.csv", "a") as f:
            writer_object = writer(f)
            writer_object.writerow([params["example_name"], bound, p])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_cores", help="Number of Cores to use")
    args = parser.parse_args()
    n_cores = int(args.n_cores)
    bounds = list(range(6, 31))

    ### Examples
    example_A = {
        "num_classes": 2,
        "num_servers": 1,
        "arrival_rates": [5, 6],
        "service_rates": [12, 13],
        "thetas": [[None, 3], [1, None]],
        "example_name": "A",
    }

    example_B = {
        "num_classes": 2,
        "num_servers": 1,
        "arrival_rates": [3, 6],
        "service_rates": [12, 13],
        "thetas": [[None, 2], [5, None]],
        "example_name": "B",
    }

    example_C = {
        "num_classes": 2,
        "num_servers": 3,
        "arrival_rates": [5, 6],
        "service_rates": [3.8, 4],
        "thetas": [[None, 1], [2, None]],
        "example_name": "C",
    }

    example_D = {
        "num_classes": 2,
        "num_servers": 2,
        "arrival_rates": [5, 6],
        "service_rates": [6.5, 7.5],
        "thetas": [[None, 2], [4, None]],
        "example_name": "D",
    }

    pool = multiprocessing.Pool(n_cores)
    func_arguments = [
        (params, bound)
        for bound in bounds
        for params in [example_A, example_B, example_C, example_D]
    ]
    random.shuffle(func_arguments)
    pool.starmap(write_row, func_arguments)
