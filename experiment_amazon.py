import numpy as np
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import pandas as pd

from fitting import fit, fit_dsl

@dataclass
class SampleParams:
    # data frame (x1, x2, x3, y, y_hat)
    data: pd.DataFrame
    # number of expert-annotated amples
    n: int
    # prediction accuracy
    pq: float

def compute_coeffs(params: SampleParams):
    """
    Generate the data and compute the coefficients for the three scenarios
    """
    selected = np.random.choice(len(params.data), size=params.n, replace=False)
    X = params.data[["x1", "x2", "x3", "x4"]].to_numpy()
    Y = params.data["y"]
    Y_hat = params.data["y_hat"]
    coeffs_all = fit(X, Y)
    coeffs_exp = fit(X[selected], Y[selected])
    coeffs_dsl = fit_dsl(X, Y, Y_hat, selected)
    return coeffs_all, coeffs_exp, coeffs_dsl

def simulate(
        # data frame (x1, x2, x3, y, y_hat)
        data: pd.DataFrame,
        # number of expert annotation to try
        num_data_points: int,
        # minimum number of expert annotations
        min_expert_samples: int,
        # prediction accuracy of the simulated LLM
        prediction_accuracy: float,
        # number of coefficients in the logistic regression
        num_coefficients: int,
        # Number of cores to parallelise over
        num_cores: int,
    ):

    # Initialise arrays
    size = (num_data_points, num_coefficients)
    coeffs_all = np.zeros(size)
    coeffs_exp = np.zeros(size)
    coeffs_dsl = np.zeros(size)

    # Generate the compute arguments
    num_expert_samples = np.logspace(
        start = np.log10(min_expert_samples), # too low = convergence issues
        stop = np.log10(len(data)),
        num = num_data_points,
        base = 10.0,
    )
    params = []
    for n in np.round(num_expert_samples).astype(int):
        params.append(SampleParams(
            data = data,
            n = n,
            pq = prediction_accuracy,
        ))

    # Compute the coefficients concurrently 
    with ProcessPoolExecutor(max_workers = num_cores) as executor:
        for i, coeffs in enumerate(executor.map(compute_coeffs, params)):
            print(f"Computed data point ({i})")
            coeffs_all[i,:] = coeffs[0]
            coeffs_exp[i,:] = coeffs[1]
            coeffs_dsl[i,:] = coeffs[2]

    return num_expert_samples, coeffs_all, coeffs_exp, coeffs_dsl


if __name__ == "__main__":

    if "SLURM_CPUS_ON_NODE" in os.environ:
        num_cores = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        num_cores = 10
    print(f"Using {num_cores} cores")

    annotated_reviews_path = Path(sys.argv[1])
    results_path = Path(sys.argv[2])

    print("Reading the data")
    data = pd.read_pickle(annotated_reviews_path)

    print("Running the experiment")
    num_expert_samples, coeffs_all, coeffs_exp, coeffs_dsl = simulate(
        data = data,
        num_data_points = 10,
        min_expert_samples = 200,
        prediction_accuracy = 0.9,
        num_coefficients = 5, # 4 Xs + intercept
        num_cores = num_cores,
    )

    print(f"Saving results to {results_path}")
    np.savez(
        results_path,
        num_expert_samples = num_expert_samples,
        coeffs_all = coeffs_all,
        coeffs_exp = coeffs_exp,
        coeffs_dsl = coeffs_dsl
    )
