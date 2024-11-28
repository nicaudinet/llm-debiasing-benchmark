import numpy as np
from dataclasses import dataclass
import scipy
from sklearn.linear_model import LogisticRegression
import tempfile
import rpy2.robjects as ro
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import sys

@dataclass
class Configs:
    """
    Configuration parameters for the experiment
    """
    # side length of the experiment grid
    side_length: int 
    # confidence level for the standard error
    confidence_level: float
    # prediction accuracy of the simulated LLM
    prediction_accuracy: float
    # number of coefficients in the logistic regression
    num_coefficients: int
    # name of the file the data will be saved in
    datafile: Path
    # Number of cores to parallelise over
    num_cores: int

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

def fit(X, Y):
    """
    Fit a logistic regression to the data and return the coefficients
    """
    logreg = LogisticRegression(fit_intercept = True)
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
class SampleConfig:
    # Total number of samples (== number of predicted samples)
    N: int
    # Number of samples for expert annotation
    n: int
    # The prediction accuracy
    pq: float

def compute_coeffs(sample: SampleConfig):
    """
    Generate the data and compute the coefficients for the three scenarios
    """
    selected = np.random.choice(sample.N, size=sample.n, replace=False)
    X, Y, Y_hat = generate(sample.N, sample.pq)
    coeffs_all = fit(X, Y)
    coeffs_exp = fit(X[selected], Y[selected])
    coeffs_dsl = fit_dsl(X, Y, Y_hat, selected)
    return coeffs_all, coeffs_exp, coeffs_dsl

def simulate(configs):

    # Initialise arrays
    size = (configs.side_length, configs.side_length, configs.num_coefficients)
    coeffs_all = np.zeros(size)
    coeffs_exp = np.zeros(size)
    coeffs_dsl = np.zeros(size)

    # Generate the compute arguments
    num_samples = np.logspace(
        start = 2,
        stop = 4,
        num = configs.side_length,
        base = 10.0,
    )
    num_samples = np.round(num_samples).astype(int)
    cells = [(i,j) for i in range(configs.side_length) for j in range(i+1)]
    samples = []
    for i, j in cells:
        sample = SampleConfig(
            N = num_samples[i],
            n = num_samples[j],
            pq = configs.prediction_accuracy,
        )
        samples.append(sample)

    # Compute the coefficients concurrently 
    with ProcessPoolExecutor(max_workers=4) as executor:
        for (i,j), coeffs in zip(cells, executor.map(compute_coeffs, samples)):
            print(f"Computed cell ({i},{j})")
            coeffs_all[i,j,:] = coeffs[0]
            coeffs_exp[i,j,:] = coeffs[1]
            coeffs_dsl[i,j,:] = coeffs[2]

    return coeffs_all, coeffs_exp, coeffs_dsl


if __name__ == "__main__":

    
    if "SLURM_CPUS_ON_NODE" in os.environ:
        num_cores = int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        num_cores = 32

#    if "SLURM_ARRAY_TASK_ID" in os.environ:
#       task_id = os.environ["SLURM_CPUS_ON_NODE"]
#       datafile = Path(f"results/data_{task_id}.npz")
#   else:
#       datafile = Path("data.npz")

    configs = Configs(
        side_length = 10,
        confidence_level = 0.95,
        prediction_accuracy = 0.9,
        num_coefficients = 5, # 4 Xs + intercept
        datafile = Path(sys.argv[1]),
        num_cores = num_cores,
    )

    coeffs_all, coeffs_exp, coeffs_dsl = simulate(configs)
    np.savez(
        configs.datafile,
        coeffs_all=coeffs_all,
        coeffs_exp=coeffs_exp,
        coeffs_dsl=coeffs_dsl
    )
