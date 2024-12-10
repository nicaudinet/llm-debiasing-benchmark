import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import os
from tqdm import tqdm 
from pathlib import Path

epsilon = 1e-5
datadir = Path("results/")
plotdir = Path("plots/")

problem_runs = [ 199, 485, 560, 762, 855, 994, ]
problem_files = [ f"data_{i}.npz" for i in problem_runs ]

print("Gathering data from files")
coeffs_all = []
coeffs_exp = []
coeffs_dsl = []
for file in tqdm(os.listdir(datadir)):
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

def plot_coeffs(coeffs, ax, norm, cbar_ax, title):
    image = coeffs.copy()
    for i in range(10):
        for j in range(10):
            if i < j:
                image[i,j] = float("NaN")
    image = np.flip(image, axis=0)
    sns.heatmap(
        image,
        ax = ax,
        cmap = "coolwarm",
        norm = norm,
        square = True,
        annot = True,
        fmt = ".2f",
        cbar_ax = cbar_ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Number of expert samples")
    ax.set_ylabel("Number of total samples")
    ax.set_xticks(ticks=[0.5, 5, 9.5], labels=["$10^2$", "$10^3$", "$10^4$"])
    ax.set_yticks(ticks=[9.5, 5, 0.5], labels=["$10^2$", "$10^3$", "$10^4$"])

rows = 1
cols = 3
fig, axs = plt.subplots(rows, cols, figsize=(cols * 10, rows * 8))
cbar_ax = fig.add_axes((.91, .15, .02, .7))
norm = LogNorm(0.4, 1.7)
plot_coeffs(coeffs_all.mean(axis=(0,3)), axs[0], norm, cbar_ax, "coeffs_all")
plot_coeffs(coeffs_exp.mean(axis=(0,3)), axs[1], norm, cbar_ax, "coeffs_exp")
plot_coeffs(coeffs_dsl.mean(axis=(0,3)), axs[2], norm, cbar_ax, "coeffs_dsl")
fig.tight_layout(rect=(0, 0, .9, 1))
plt.savefig(plotdir / Path("coeffs_mean.svg"))

def calc_bias(Y, Y_hat):
    error_std = (Y - Y_hat) # / np.maximum(epsilon, Y)
    return error_std.mean(axis=(0,3))

rows = 1
cols = 3
fig, axs = plt.subplots(rows, cols, figsize=(cols * 10, rows * 8))
cbar_ax = fig.add_axes((.91, .15, .02, .7))
norm = Normalize(-1.25, 1.25)
bias_exp = calc_bias(coeffs_all, coeffs_exp)
bias_dsl = calc_bias(coeffs_all, coeffs_dsl)
error = bias_exp - bias_dsl
plot_coeffs(bias_exp, axs[0], norm, cbar_ax, "Bias_exp = coeffs_all - coeffs_exp")
plot_coeffs(bias_dsl, axs[1], norm, cbar_ax, "Bias_dsl = coeffs_all - coeffs_dsl")
plot_coeffs(error, axs[2], norm, cbar_ax, "Error = bias_exp - bias_dsl")
fig.tight_layout(rect=(0, 0, .9, 1))
plt.savefig(plotdir / Path("bias.svg"))

def calc_rmse(Y, Y_hat):
    return np.sqrt(np.mean(np.square(Y - Y_hat), axis=(0,3)))

rows = 1
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(cols * 10, rows * 8))
cbar_ax = fig.add_axes((.91, .15, .02, .7))
norm = Normalize()
plot_coeffs(calc_rmse(coeffs_all, coeffs_exp), axs[0], norm, cbar_ax, "RMSE(all,exp)")
plot_coeffs(calc_rmse(coeffs_all, coeffs_dsl), axs[1], norm, cbar_ax, "RMSE(all,dsl)")
fig.tight_layout(rect=(0, 0, .9, 1))
plt.savefig(plotdir / Path("rmse.svg"))
