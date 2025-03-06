import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from argparse import ArgumentParser

###################
# Parse arguments #
###################

parser = ArgumentParser()
parser.add_argument("results_path", type = Path)
parser.add_argument("plot_dir", type = Path)
args = parser.parse_args()

###############
# Gather data #
###############

print("Gathering data from files")
coeffs_all = []
coeffs_exp = []
coeffs_dsl = []
coeffs_ppi = []

files = [args.results_path / Path(f) for f in os.listdir(args.results_path)]
files = [f for f in files if os.path.isfile(f)]
if len(files) == 0:
    raise Exception(f"No data files found in results dir: {args.results_path}")

for file in files:
    data = np.load(file)
    coeffs_all.append(data["coeffs_all"])
    coeffs_exp.append(data["coeffs_exp"])
    coeffs_dsl.append(data["coeffs_dsl"])
    coeffs_ppi.append(data["coeffs_ppi"])

coeffs_all = np.stack(coeffs_all, axis=0)
coeffs_exp = np.stack(coeffs_exp, axis=0)
coeffs_dsl = np.stack(coeffs_dsl, axis=0)
coeffs_ppi = np.stack(coeffs_ppi, axis=0)

total_reps = coeffs_all.shape[0]
print(f"Total files: {total_reps}")

# Assume X-axis (number of expert samples) is the same for all files
X = data["num_expert_samples"]

################
# Compute RMSE #
################

# Compute RMSE with all
def compute_rmse(coeffs_true, coeffs_pred):
    assert coeffs_true.shape[0] == coeffs_pred.shape[0]
    rmse = np.sqrt(np.mean((coeffs_true - coeffs_pred) ** 2, axis=(0,2)))
    sd = np.sqrt(np.mean((coeffs_true - coeffs_pred) ** 2, axis=2)).std(axis=0)
    num_repetitions = coeffs_true.shape[0]
    std_err = sd / np.sqrt(num_repetitions)
    upper = rmse + 2 * std_err
    lower = rmse - 2 * std_err
    return rmse, upper, lower

rmse_exp, upper_exp, lower_exp = compute_rmse(coeffs_all, coeffs_exp)
rmse_dsl, upper_dsl, lower_dsl = compute_rmse(coeffs_all, coeffs_dsl)
rmse_ppi, upper_ppi, lower_ppi = compute_rmse(coeffs_all, coeffs_ppi)

#############
# Plot RMSE #
#############

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

print(coeffs_all.shape)

plt.figure()
plt.xscale('log')
plt.title(f"Varying the number of expert samples (R = {total_reps})")
plt.xlabel("Number of expert samples (n)")
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

plt.fill_between(
    X,
    lower_ppi,
    upper_ppi,
    color = colors[2],
    alpha = 0.2,
    linewidth = 0,
)
plt.plot(X, rmse_ppi, "o-", color = colors[2], label = "PPI")

plt.legend()

plt.savefig(args.plot_dir / Path(f"rmse.pdf"))
plt.savefig(args.plot_dir / Path(f"rmse.png"))
