import numpy as np
import scipy
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import sys

from fitting import fit, fit_dsl

def generate(num_samples, prediction_accuracy):
    """
    Generate the simulated data (following Appendix E)
    """
    # number of covariates
    M = 10

    # Generate the covariates
    means = np.zeros(M)
    covs = 0.3 * np.ones((M, M))
    covs = covs + 0.7 * np.eye(M)
    X = np.random.multivariate_normal(means, covs, size=num_samples)

    # Binarize X_i2
    X[:,1] = (X[:,1] > scipy.stats.norm.ppf(0.8)).astype(np.float32)

    # Generate the true outcome Y as a function of the covariates
    X1, X2, X3, X4, X6 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,5]
    a = 0.1 / (1.0 + np.exp(0.5 * X3 - 0.5 * X2))
    b = 1.3 * X4 / (1.0 + np.exp(-0.1 * X2))
    c = 1.5 * X4 * X6
    d = 0.5 * X1 * X2
    e = 1.3 * X1
    f = X2
    W = a + b + c + d + e + f
    Y_true = np.random.binomial(1, scipy.special.expit(W), size=num_samples)
    Y_true = Y_true.astype(float)

    # Make training data from true covariates
    X_train = np.stack([
        X[:,0],
        X[:,0] ** 2,
        X[:,1],
        X[:,3],
    ], axis=1)

    # Simulate the LLM annotations
    p = np.random.binomial(1, prediction_accuracy, size=num_samples)
    Y_pred = p * Y_true + (1 - p) * (1 - Y_true)

    return X_train, Y_true, Y_pred

@dataclass
class SampleParams:
    # total number of samples
    N: int
    # number of expert-annotated amples
    n: int
    # prediction accuracy
    pq: float

def compute_coeffs(params: SampleParams):
    """
    Generate the data and compute the coefficients for the three scenarios
    """
    selected = np.random.choice(params.N, size=params.n, replace=False)
    X, Y, Y_hat = generate(params.N, params.pq)
    coeffs_all = fit(X, Y)
    coeffs_exp = fit(X[selected], Y[selected])
    coeffs_dsl = fit_dsl(X, Y, Y_hat, selected)
    return coeffs_all, coeffs_exp, coeffs_dsl


def simulate(
        # number of expert samples
        num_expert_samples: int,
        # number of expert annotation to try
        num_data_points: int,
        # maximum number of expert annotations
        max_total_samples: int,
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
    num_total_samples = np.logspace(
        start = np.log10(num_expert_samples), # too low = convergence issues
        stop = np.log10(max_total_samples),
        num = num_data_points,
        base = 10.0,
    )
    params = []
    for N in np.round(num_total_samples).astype(int):
        params.append(SampleParams(
            N = N,
            n = num_expert_samples,
            pq = prediction_accuracy,
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
        num_cores = 11
    print(f"Using {num_cores} cores")

    datafile = Path(sys.argv[1])
    n = int(sys.argv[2])

    num_total_samples, coeffs_all, coeffs_exp, coeffs_dsl = simulate(
        num_expert_samples = n,
        num_data_points = 10,
        max_total_samples = 10000,
        prediction_accuracy = 0.9,
        num_coefficients = 5, # 4 Xs + intercept
        num_cores = num_cores,
    )

    print(f"Saving results to {datafile}")
    np.savez(
        datafile,
        num_total_samples = num_total_samples,
        coeffs_all = coeffs_all,
        coeffs_exp = coeffs_exp,
        coeffs_dsl = coeffs_dsl
    )
