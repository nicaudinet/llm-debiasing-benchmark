import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
import tempfile
import rpy2.robjects as ro
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import sys

from generate import generate

def fit(X, Y):
    """
    Fit a logistic regression to the data and return the coefficients
    """
    logreg = LogisticRegression(
        fit_intercept = True, # also fit the intercept
        penalty = None, # no regularisation (default in R)
    )
    logreg.fit(X, Y)
    coeffs = np.insert(logreg.coef_, 0, logreg.intercept_)
    assert len(coeffs) == 5
    return coeffs

def fit_dsl(X, Y_true, Y_pred, selected):
    """
    Fit a logistic regression to the data with predicted and expert annotations
    using the DSL from the R package
    """

    # Set Y_true to None for non-selected samples
    not_selected = set(range(Y_true.shape[0])) - set(selected)
    not_selected = list(not_selected)
    Y_true_sel = Y_true.copy()
    Y_true_sel[not_selected] = None # selected expert annotations

    # Put the data into a dataframe
    data = pd.DataFrame({
        "c0": X[:,0],
        "c1": X[:,1],
        "c2": X[:,2],
        "c3": X[:,3],
        "Y": Y_true_sel,
        "Y_hat": Y_pred,
    })

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp:

        # Create temporary files
        data_file = Path(tmp) / "data.csv"
        coeff_file = Path(tmp) / "coeff.csv"

        # Write data to file
        data.to_csv(data_file, index=False)

        # R script that reads the data file, runs the DSL algorithm and
        # writes the coefficients to the result file
        ro.r(f"""
            sink("/dev/null")
            library("dsl")
            data <- read.csv("{data_file}")
            out <- dsl(
                model = "logit",
                formula = Y ~ c0 + c1 + c2 + c3,
                predicted_var = "Y",
                prediction = "Y_hat",
                data = data,
                seed = Sys.time()
            )
            write.csv(out$coefficients, "{coeff_file}", row.names=FALSE)
            sink()
        """)

        # Read coefficients from the result file
        coeffs = np.array(pd.read_csv(coeff_file)).squeeze()
        assert len(coeffs) == 5

    return coeffs

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
        # total number of samples
        num_total_samples: int,
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
        stop = np.log10(num_total_samples),
        num = num_data_points,
        base = 10.0,
    )
    params = []
    for n in np.round(num_expert_samples).astype(int):
        params.append(SampleParams(
            N = num_total_samples,
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

    num_expert_samples, coeffs_all, coeffs_exp, coeffs_dsl = simulate(
        num_total_samples = 10000,
        num_data_points = 3,
        min_expert_samples = 200,
        prediction_accuracy = 0.9,
        num_coefficients = 5, # 4 Xs + intercept
        num_cores = num_cores,
    )

    datafile = Path(sys.argv[1])
    print(f"Saving results to {datafile}")
    np.savez(
        datafile,
        num_expert_samples = num_expert_samples,
        coeffs_all = coeffs_all,
        coeffs_exp = coeffs_exp,
        coeffs_dsl = coeffs_dsl
    )
