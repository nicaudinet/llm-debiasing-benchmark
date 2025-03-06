import numpy as np
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from argparse import ArgumentParser

from fitting import fit, fit_dsl, fit_ppi

@dataclass
class SampleParams:
    data: pd.DataFrame
    n: int

def compute_coeffs(params: SampleParams):

    selected = np.random.choice(len(params.data), size=params.n, replace=False)

    X = params.data[["x1", "x2", "x3", "x4"]].to_numpy()
    Y = params.data["y"].to_numpy().astype(float)
    Y_hat = params.data["y_hat"].to_numpy()

    coeffs_all = fit(X, Y)
    coeffs_exp = fit(X[selected], Y[selected])
    coeffs_dsl = fit_dsl(X, Y, Y_hat, selected)
    coeffs_ppi = fit_ppi(X, Y, Y_hat, selected)

    return coeffs_all, coeffs_exp, coeffs_dsl, coeffs_ppi

def simulate(
        data: pd.DataFrame, # expected columns: (x1, x2, x3, y, y_hat)
        num_data_points: int,
        min_expert_samples: int,
        num_coefficients: int,
        num_cores: int,
    ):

    size = (num_data_points, num_coefficients)
    results = {
        "coeffs_all": np.zeros(size),
        "coeffs_exp": np.zeros(size),
        "coeffs_dsl": np.zeros(size),
        "coeffs_ppi": np.zeros(size),
    }

    num_expert_samples = np.logspace(
        start = np.log10(min_expert_samples), # too low = convergence issues
        stop = np.log10(len(data)),
        num = num_data_points,
        base = 10.0,
    )
    results["num_expert_samples"] = num_expert_samples

    params = []
    for n in np.round(num_expert_samples).astype(int):
        params.append(SampleParams(
            data = data,
            n = n,
        ))

    with ProcessPoolExecutor(max_workers = num_cores) as executor:
        for i, coeffs in enumerate(executor.map(compute_coeffs, params)):
            print(f"Computed data point ({i})")
            results["coeffs_all"][i,:] = coeffs[0]
            results["coeffs_exp"][i,:] = coeffs[1]
            results["coeffs_dsl"][i,:] = coeffs[2]
            results["coeffs_ppi"][i,:] = coeffs[3]

    return results


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("annotated_path", type = Path)
    parser.add_argument("results_path", type = Path)
    parser.add_argument("--seed", type = int)
    args = parser.parse_args()

    if "SLURM_CPUS_ON_NODE" in os.environ:
        num_cores = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        num_cores = 11
    print(f"Using {num_cores} cores")

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using seed = {args.seed}")
    else:
        print("The seed was not provided, using current system time")

    print("Reading the data")
    data = pd.read_json(args.annotated_path)

    print("Running the experiment")
    results = simulate(
        data = data,
        num_data_points = 10,
        min_expert_samples = 200,
        num_coefficients = 5, # 4 Xs + intercept
        num_cores = num_cores,
    )

    print(f"Saving results to {args.results_path}")
    np.savez(args.results_path, **results)
