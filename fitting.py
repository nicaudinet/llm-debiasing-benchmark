import numpy as np
from sklearn.linear_model import LogisticRegression
import tempfile
import rpy2.robjects as ro
from pathlib import Path
import pandas as pd

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

    return coeffs
