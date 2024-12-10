import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-5
datafile = "data.npz"
plotfile = "plot.pdf"

def calc_bias(Y, Y_hat):
    error_std = (Y - Y_hat) / np.maximum(epsilon, Y)
    bias = error_std.mean(axis=2)
    assert bias.shape == (3,3)
    return bias

data = np.load(datafile)
bias_exp = calc_bias(data["coeffs_all"], data["coeffs_exp"])
bias_dsl = calc_bias(data["coeffs_all"], data["coeffs_dsl"])
error = bias_exp / np.maximum(epsilon, bias_dsl)
error = np.flip(error, axis=0) # flip along the horizontal axis for the plot

plt.figure()
sns.heatmap(error, cmap = "coolwarm", annot = True, fmt = ".2f")
plt.title("Error")
plt.xlabel("Number of expert samples")
plt.ylabel("Number of total samples")
plt.xticks(ticks=[0.5, 5, 9.5], labels=["$10^2$", "$10^3$", "$10^4$"])
plt.yticks(ticks=[9.5, 5, 0.5], labels=["$10^2$", "$10^3$", "$10^4$"])
plt.savefig(plotfile)
