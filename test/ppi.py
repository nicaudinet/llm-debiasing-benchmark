import scipy
import numpy as np
from ppi_py import ppi_logistic_pointestimate, ppi_logistic_ci

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

if __name__ == "__main__":

    np.random.seed(0)

    alpha = 0.1 # Error rate
    N = 5000
    n = 500
    assert n <= N

    print("Generating the data")
    X_total, Y_total, Yhat_total = generate(N, 0.9)

    # Add intercept
    ones = np.ones((X_total.shape[0], 1))
    X_total = np.concatenate([ones, X_total], axis=1)

    rand_idx = np.random.permutation(Y_total.shape[0])
    labelled = rand_idx[:n]
    unlabelled = rand_idx[n:]

    print("Computing PPI point estimate")
    ppi_pe = ppi_logistic_pointestimate(
        X = X_total[labelled],
        Y = Y_total[labelled],
        Yhat = Yhat_total[labelled],
        X_unlabeled = X_total[unlabelled],
        Yhat_unlabeled = Yhat_total[unlabelled],
    )

    print("Computing PPI confidence intervals")
    ppi_ci = ppi_logistic_ci(
        X = X_total[labelled],
        Y = Y_total[labelled],
        Yhat = Yhat_total[labelled],
        X_unlabeled = X_total[unlabelled],
        Yhat_unlabeled = Yhat_total[unlabelled],
        alpha = alpha,
    )

    print("Results:")
    for i in range(len(ppi_pe)):
        print(f" - Coeff {i}: {ppi_pe[i]} ({ppi_ci[0][i]}, {ppi_ci[1][i]})")
