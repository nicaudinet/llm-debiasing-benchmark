import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import os
from tqdm import tqdm 
from pathlib import Path

datadir = Path("results/run2")
plotdir = Path("results/run2/plots/")

problem_runs = [] # [ 199, 485, 560, 762, 855, 994, ]
problem_files = [ f"data_{i}.npz" for i in problem_runs ]

print("Gathering data from files")
coeffs_all = []
coeffs_exp = []
coeffs_dsl = []
files = (f for f in os.listdir(datadir) if os.path.isfile(datadir / Path(f)))
for file in tqdm(files):
    if file in problem_files:
        continue
    data = np.load(datadir / Path(file))
    coeffs_all.append(data["coeffs_all"])
    coeffs_exp.append(data["coeffs_exp"])
    coeffs_dsl.append(data["coeffs_dsl"])
coeffs_all = np.stack(coeffs_all, axis=0)
coeffs_exp = np.stack(coeffs_exp, axis=0)
coeffs_dsl = np.stack(coeffs_dsl, axis=0)
print(f"Total files: {coeffs_all.shape[0]}")

# Only keep samples with N = 10000 (total samples)
coeffs_all = coeffs_all[:, -1, :, :]
coeffs_exp = coeffs_exp[:, -1, :, :]
coeffs_dsl = coeffs_dsl[:, -1, :, :]

print(coeffs_all.shape)
print(coeffs_exp.shape)
print(coeffs_dsl.shape)

# Compute RMSE with all
rmse_exp = np.sqrt(np.mean((coeffs_all - coeffs_exp) ** 2, axis=(0,2)))
rmse_dsl = np.sqrt(np.mean((coeffs_all - coeffs_dsl) ** 2, axis=(0,2)))

assert rmse_exp.shape[0] == coeffs_exp.shape[1]
assert rmse_dsl.shape[0] == coeffs_dsl.shape[1]

# Bounds for exp
rmse_exp_sd = np.sqrt(np.mean((coeffs_all - coeffs_exp) ** 2, axis=2)).std(axis=0)
upper_exp = rmse_exp + 2 * rmse_exp_sd
lower_exp = rmse_exp - 2 * rmse_exp_sd

rmse_dsl_sd = np.sqrt(np.mean((coeffs_all - coeffs_dsl) ** 2, axis=2)).std(axis=0)
upper_dsl = rmse_dsl + 2 * rmse_dsl_sd
lower_dsl = rmse_dsl - 2 * rmse_dsl_sd

# Expert sample dimentions
X = np.logspace(
    start = np.log10(200), # too low = convergence issues
    stop = np.log10(10000),
    num = 10,
    base = 10.0,
)
X = np.round(X).astype(int)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure()
plt.xscale('log')
plt.title("DSL vs. only expert-annotated samples (N = 10000)")
plt.xlabel("Number of expert-annotated samples")
plt.ylabel("RMSE")

plt.fill_between(X, lower_exp, upper_exp, color = colors[0], alpha = 0.2)
plt.plot(X, rmse_exp, "o-", color = colors[0], label = "RMSE(all,exp)")

plt.fill_between(X, lower_dsl, upper_dsl, color = colors[1], alpha = 0.2)
plt.plot(X, rmse_dsl, "o-", color = colors[1], label = "RMSE(all,dsl)")

plt.legend()

plt.savefig(plotdir / Path("rmse_1D.pdf"))
plt.savefig(plotdir / Path("rmse_1D.png"))
plt.show()
