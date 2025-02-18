import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

datadir = Path(sys.argv[1])
plotdir = Path(sys.argv[2])
n = sys.argv[3]

print("Gathering data from files")
coeffs_all = []
coeffs_exp = []
coeffs_dsl = []
files = [f for f in os.listdir(datadir) if os.path.isfile(datadir / Path(f))]
for file in files:
    data = np.load(datadir / Path(file))
    coeffs_all.append(data["coeffs_all"])
    coeffs_exp.append(data["coeffs_exp"])
    coeffs_dsl.append(data["coeffs_dsl"])
coeffs_all = np.stack(coeffs_all, axis=0)
coeffs_exp = np.stack(coeffs_exp, axis=0)
coeffs_dsl = np.stack(coeffs_dsl, axis=0)
print(f"Total files: {coeffs_all.shape[0]}")

# Assume X-axis (number of expert samples) is the same for all files
data = np.load(datadir / Path(files[0]))
X = data["num_total_samples"]

# Compute RMSE with all
rmse_exp = np.sqrt(np.mean((coeffs_all - coeffs_exp) ** 2, axis=(0,2)))
rmse_dsl = np.sqrt(np.mean((coeffs_all - coeffs_dsl) ** 2, axis=(0,2)))

assert rmse_exp.shape[0] == coeffs_exp.shape[1]
assert rmse_dsl.shape[0] == coeffs_dsl.shape[1]

rmse_exp_sd = np.sqrt(np.mean((coeffs_all - coeffs_exp) ** 2, axis=2)).std(axis=0)
upper_exp = rmse_exp + 2 * rmse_exp_sd
lower_exp = rmse_exp - 2 * rmse_exp_sd

rmse_dsl_sd = np.sqrt(np.mean((coeffs_all - coeffs_dsl) ** 2, axis=2)).std(axis=0)
upper_dsl = rmse_dsl + 2 * rmse_dsl_sd
lower_dsl = rmse_dsl - 2 * rmse_dsl_sd

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

print(coeffs_all.shape)

plt.figure()
plt.xscale('log')
plt.title(f"Varying the number of total samples (n = {n})")
plt.xlabel("Number of total samples (N)")
plt.ylabel("RMSE w.r.t. gold annotations for all samples")

plt.fill_between(
    X,
    lower_exp,
    upper_exp,
    color = colors[0],
    alpha = 0.2,
    linewidth = 0,
)
plt.plot(X, rmse_exp, "o-", color = colors[0], label = "expert only")

plt.fill_between(
    X,
    lower_dsl,
    upper_dsl,
    color = colors[1],
    alpha = 0.2,
    linewidth = 0,
)
plt.plot(X, rmse_dsl, "o-", color = colors[1], label = "DSL")

plt.legend()

plt.savefig(plotdir / Path(f"rmse_n{n}.pdf"))
plt.savefig(plotdir / Path(f"rmse_n{n}.png"))
