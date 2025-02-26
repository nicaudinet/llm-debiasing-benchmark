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
    data: pd.DataFrame # (x1, x2, x3, y, y_hat)
    N: int
    n: int

def compute_coeffs(params: SampleParams):
    """
    Generate the data and compute the coefficients for the three scenarios
    """

    X = params.data[["x1", "x2", "x3", "x4"]].to_numpy()
    Y = params.data["y"].to_numpy().astype(float)
    Y_hat = params.data["y_hat"].to_numpy()
    coeffs_all = fit(X, Y)

    samples = np.random.choice(len(params.data), size=params.N, replace=False)
    selected = np.random.choice(params.N, size=params.n, replace=False)
    X = X[samples]
    Y = Y[samples]
    Y_hat = Y_hat[samples]

    coeffs_exp = fit(X[selected], Y[selected])
    coeffs_dsl = fit_dsl(X, Y, Y_hat, selected)

    return coeffs_all, coeffs_exp, coeffs_dsl

def simulate(
        data: pd.DataFrame,
        num_expert_samples: int,
        num_data_points: int,
        max_total_samples: int,
        num_coefficients: int,
        num_cores: int,
    ):

    # Initialise arrays
    size = (num_data_points, num_coefficients)
    coeffs_all = np.zeros(size)
    coeffs_exp = np.zeros(size)
    coeffs_dsl = np.zeros(size)

    # Generate the compute arguments
    num_total_samples = np.logspace(
        start = np.log10(num_expert_samples), # too low = convergence issues
        stop = np.log10(max_total_samples),
        num = num_data_points,
        base = 10.0,
    )
    params = []
    for N in np.round(num_total_samples).astype(int):
        params.append(SampleParams(
            data = data,
            N = N,
            n = num_expert_samples,
        ))

    # Compute the coefficients concurrently 
    with ProcessPoolExecutor(max_workers = num_cores) as executor:
        for i, coeffs in enumerate(executor.map(compute_coeffs, params)):
            print(f"Computed data point ({i})")
            coeffs_all[i,:] = coeffs[0]
            coeffs_exp[i,:] = coeffs[1]
            coeffs_dsl[i,:] = coeffs[2]

    return num_total_samples, coeffs_all, coeffs_exp, coeffs_dsl


if __name__ == "__main__":

    if "SLURM_CPUS_ON_NODE" in os.environ:
        num_cores = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        num_cores = 10
    print(f"Using {num_cores} cores")

    annotated_data_path = Path(sys.argv[1])
    results_path = Path(sys.argv[2])
    num_expert_samples = int(sys.argv[3])

    try:
        seed = np.random.seed(int(sys.argv[4]))
        print(f"Using seed = {seed}")
    except Exception:
        print("The seed was not provided, using current system time")

    print("Reading the data")
    data = pd.read_pickle(annotated_data_path)
    assert 10000 <= len(data)

    print("Running the experiment")
    num_total_samples, coeffs_all, coeffs_exp, coeffs_dsl = simulate(
        data = data,
        num_expert_samples = num_expert_samples,
        num_data_points = 10,
        max_total_samples = 10000,
        num_coefficients = 5, # 4 Xs + intercept
        num_cores = num_cores,
    )

    print(f"Saving results to {results_path}")
    np.savez(
        results_path,
        num_total_samples = num_total_samples,
        coeffs_all = coeffs_all,
        coeffs_exp = coeffs_exp,
        coeffs_dsl = coeffs_dsl
    )
