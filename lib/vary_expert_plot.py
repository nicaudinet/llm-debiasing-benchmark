import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from argparse import ArgumentParser

###################
# Parse arguments #
###################

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument("results_path", type = Path)
    parser.add_argument("plot_dir", type = Path)
    parser.add_argument(
        "--norm",
        choices = ["raw", "per-coeff", "odds", "logodds", "percent"],
        default = "symlog",
    )
    return parser.parse_args()

###############
# Gather data #
###############

def gather(dataset, annotation):

    print(f" - {dataset}/{annotation}")

    base_path = Path("/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/vary-num-expert")
    data_path = base_path / dataset / "data" / annotation

    if not data_path.exists():
        raise Exception(f"Error: {data_path} not found")

    try:
        next(data_path.iterdir())
    except StopIteration:
        raise Exception(f"Error: {data_path} is empty")

    coeffs_all = []
    coeffs_exp = []
    coeffs_dsl = []
    coeffs_ppi = []

    for file in data_path.iterdir():
        data = np.load(file)
        coeffs_all.append(data["coeffs_all"])
        coeffs_exp.append(data["coeffs_exp"])
        coeffs_dsl.append(data["coeffs_dsl"])
        coeffs_ppi.append(data["coeffs_ppi"])

    return {
        "num_expert_samples": data["num_expert_samples"],
        "all": np.stack(coeffs_all, axis=0),
        "exp": np.stack(coeffs_exp, axis=0),
        "dsl": np.stack(coeffs_dsl, axis=0),
        "ppi": np.stack(coeffs_ppi, axis=0),
    }

###########
# Metrics #
###########

def compute_bias(coeffs_true, coeffs_pred):

    assert coeffs_true.shape == coeffs_pred.shape

    R = coeffs_true.shape[0]

    error = (coeffs_true - coeffs_pred) / coeffs_true
    bias = np.mean(error, axis = (0,2))

    std_err = np.std(error, axis = (0,2)) / np.sqrt(R)
    upper = bias + 2 * std_err
    lower = bias - 2 * std_err
    return bias, upper, lower

def compute_rmse(coeffs_true, coeffs_pred, standardise):

    assert len(coeffs_true.shape) == 3
    assert coeffs_true.shape == coeffs_pred.shape

    R = coeffs_true.shape[0]

    if standardise:
        # standardsize per coeff
        error = (coeffs_true - coeffs_pred) / coeffs_true
    else:
        error = coeffs_true - coeffs_pred

    rmse = np.sqrt(np.mean(error ** 2, axis=(0,2)))

    sd = np.sqrt(np.mean(error ** 2, axis=2)).std(axis=0)
    std_err = sd / np.sqrt(R)
    upper = rmse + 2 * std_err
    lower = rmse - 2 * std_err

    return {
        "rmse": rmse,
        "upper": upper,
        "lower": lower,
    }

def compute_rmse_ratio(coeffs_all, coeffs_exp, coeffs_dsl, coeffs_ppi, ratio):

    assert len(coeffs_all.shape) == 3
    assert coeffs_all.shape == coeffs_exp.shape == coeffs_dsl.shape == coeffs_ppi.shape

    R = coeffs_all.shape[0]

    rmse_exp = np.sqrt(np.mean((coeffs_all - coeffs_exp) ** 2, axis=2))
    rmse_dsl = np.sqrt(np.mean((coeffs_all - coeffs_dsl) ** 2, axis=2))
    rmse_ppi = np.sqrt(np.mean((coeffs_all - coeffs_ppi) ** 2, axis=2))

    if ratio == "odds":
        rmse_dsl = rmse_dsl / rmse_exp
        rmse_ppi = rmse_ppi / rmse_exp
    elif ratio == "logodds":
        rmse_dsl = np.log(rmse_dsl / rmse_exp)
        rmse_ppi = np.log(rmse_ppi / rmse_exp)
    elif ratio == "percent":
        rmse_dsl = 0.5 * (rmse_dsl - rmse_exp) / (rmse_dsl + rmse_exp)
        rmse_ppi = 0.5 * (rmse_ppi - rmse_exp) / (rmse_ppi + rmse_exp)
    else:
        raise Exception(f"ratio {ratio} not recognised")

    sd_dsl = rmse_dsl.std(axis=0)
    rmse_dsl = rmse_dsl.mean(axis=0)
    dsl = {
        "rmse": rmse_dsl,
        "upper": rmse_dsl + 2 * sd_dsl / np.sqrt(R),
        "lower": rmse_dsl - 2 * sd_dsl / np.sqrt(R),
    }

    sd_ppi = rmse_ppi.std(axis=0)
    rmse_ppi = rmse_ppi.mean(axis=0)
    ppi = {
        "rmse": rmse_ppi,
        "upper": rmse_ppi + 2 * sd_ppi / np.sqrt(R),
        "lower": rmse_ppi - 2 * sd_ppi / np.sqrt(R),
    }

    return dsl, ppi

#############
# Plot RMSE #
#############

# Forward 
def forward(x, N):
    """
    Transformation from linspace(0,1) to logspace(log(200),log(N))/N
    """
    return N**(x-1) * 200**(1-x)

def inverse(x, N):
    """
    Transformation from logspace(log(200),log(N))/N to linspace(0,1) to 
    """
    return 1 + np.log10(x) / np.log10(N / 200)


def plot_bias():
    pass

# print(" - Plotting the bias")
#
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# plt.figure()
# plt.xscale('log')
# plt.title(f"Standardised bias vs. number of expert samples (R = {total_reps})")
# plt.xlabel("Number of expert samples (n)")
# plt.ylabel("Standardised bias")
#
# plt.fill_between(
#     X,
#     bias_lower_exp,
#     bias_upper_exp,
#     color = colors[0],
#     alpha = 0.2,
#     linewidth = 0,
# )
# plt.plot(X, bias_exp, "o-", color = colors[0], label = "expert only")
#
# plt.fill_between(
#     X,
#     bias_lower_dsl,
#     bias_upper_dsl,
#     color = colors[1],
#     alpha = 0.2,
#     linewidth = 0,
# )
# plt.plot(X, bias_dsl, "o-", color = colors[1], label = "DSL")
#
# plt.fill_between(
#     X,
#     bias_lower_ppi,
#     bias_upper_ppi,
#     color = colors[2],
#     alpha = 0.2,
#     linewidth = 0,
# )
# plt.plot(X, bias_ppi, "o-", color = colors[2], label = "PPI")
#
# plt.legend()
#
# plt.savefig(args.plot_dir / Path(f"bias.pdf"))
# plt.savefig(args.plot_dir / Path(f"bias.png"))


def plot_rmse(ax, exp, dsl, ppi, standardise):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    X = np.linspace(0, 1, 10)

    ax.fill_between(
        X,
        exp["lower"],
        exp["upper"],
        color = colors[0],
        alpha = 0.2,
        linewidth = 0,
    )
    ax.plot(
        X,
        exp["rmse"],
        "o-",
        color = colors[0],
        label = "expert-only",
    )

    ax.fill_between(
        X,
        dsl["lower"],
        dsl["upper"],
        color = colors[1],
        alpha = 0.2,
        linewidth = 0,
    )
    ax.plot(
        X,
        dsl["rmse"],
        "o-",
        color = colors[1],
        label = "DSL",
    )

    ax.fill_between(
        X,
        ppi["lower"],
        ppi["upper"],
        color = colors[2],
        alpha = 0.2,
        linewidth = 0,
    )
    ax.plot(
        X,
        ppi["rmse"],
        "o-",
        color = colors[2],
        label = "PPI",
    )

    if standardise:
        ax.set_ylabel("Standardised RMSE")
    else:
        ax.set_ylabel("RMSE")

    xticklabels = [f"{x:.2f}" for x in forward(X, 10000)]
    ax.set_xticks(ticks = X, labels = xticklabels)
    ax.set_xlabel("Proportion of expert samples (log)")
    ax.legend()

def plot_rmse_ratio(ax, dsl, ppi, ratio):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    X = np.linspace(0, 1, 10)

    ax.fill_between(
        X,
        dsl["lower"],
        dsl["upper"],
        color = colors[1],
        alpha = 0.2,
        linewidth = 0,
    )
    ax.plot(
        X,
        dsl["rmse"],
        "o-",
        color = colors[1],
        label = "DSL",
    )

    ax.fill_between(
        X,
        ppi["lower"],
        ppi["upper"],
        color = colors[2],
        alpha = 0.2,
        linewidth = 0,
    )
    ax.plot(
        X,
        ppi["rmse"],
        "o-",
        color = colors[2],
        label = "PPI",
    )

    if ratio == "odds":
        ax.set_ylabel("RMSE Odds Ratio")
        ax.axhline(1, linestyle = "--", color = "grey", alpha = 0.2)
        ax.set_yscale("log")
    elif ratio == "logodds":
        ax.set_ylabel("RMSE Log Odds Ratio")
        ax.axhline(0, linestyle = "--", color = "grey", alpha = 0.2)
    elif ratio == "percent":
        ax.set_ylabel("RMSE Symmetric Percent Change")
        ax.axhline(0, linestyle = "--", color = "grey", alpha = 0.2)
    else:
        raise Exception(f"ratio {ratio} not supported in plot_rmse_ratio")

    xticklabels = [f"{x:.2f}" for x in forward(X, 10000)]
    ax.set_xticks(ticks = X, labels = xticklabels)
    ax.set_xlabel("Proportion of expert samples (log)")
    ax.legend()
    

def plot_all(ax, data, norm):

    R = min(data[d][a]["all"].shape[0] for d in datasets for a in annotations)
    print(f"- minimum number of repetitions: {R}")

    D = len(datasets)
    A = len(annotations)
    num_exp = np.zeros((D, A, 10))
    size = (D, A, R, 10, 5)
    coeffs_all = np.zeros(size)
    coeffs_exp = np.zeros(size)
    coeffs_dsl = np.zeros(size)
    coeffs_ppi = np.zeros(size)

    for i, d in enumerate(datasets):
        for j, a in enumerate(annotations):
            N = data[d][a]["all"].shape[0]
            subsample = np.random.choice(N, R, replace = False)
            coeffs_all[i,j,:,:,:] = data[d][a]["all"][subsample,:,:]
            coeffs_exp[i,j,:,:,:] = data[d][a]["exp"][subsample,:,:]
            coeffs_dsl[i,j,:,:,:] = data[d][a]["dsl"][subsample,:,:]
            coeffs_ppi[i,j,:,:,:] = data[d][a]["ppi"][subsample,:,:]
            num_exp[i,j,:] = data[d][a]["num_expert_samples"]

    transposed = (2, 3, 4, 0, 1)
    reshaped = (R, 10, 5 * D * A)
    coeffs_all = coeffs_all.transpose(transposed).reshape(reshaped)
    coeffs_exp = coeffs_exp.transpose(transposed).reshape(reshaped)
    coeffs_dsl = coeffs_dsl.transpose(transposed).reshape(reshaped)
    coeffs_ppi = coeffs_ppi.transpose(transposed).reshape(reshaped)

    print("- computing and plotting the RMSE")
    if norm == "raw":
        print("- using no normalisation")
        rmse_exp = compute_rmse(coeffs_all, coeffs_exp, False)
        rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl, False)
        rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi, False)
        plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi, False)
    elif norm == "per-coeff":
        print("- using per-coefficient normalisation")
        rmse_exp = compute_rmse(coeffs_all, coeffs_exp, True)
        rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl, True)
        rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi, True)
        plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi, True)
    elif norm == "odds" or norm == "logodds" or norm == "percent":
        print(f"- using ratio ({norm}) normalisation")
        rmse_dsl, rmse_ppi = compute_rmse_ratio(
            coeffs_all,
            coeffs_exp,
            coeffs_dsl,
            coeffs_ppi,
            norm,
        )
        plot_rmse_ratio(ax, rmse_dsl, rmse_ppi, norm)
    else:
        raise Exception(f"norm {norm} not supported in plot_all")


def plot_dataset(ax, data, norm):

    R = min(data[a]["all"].shape[0] for a in annotations)
    print(f"- minimum number of repetitions: {R}")

    A = len(annotations)
    num_exp = np.zeros((A, 10))
    size = (A, R, 10, 5)
    coeffs_all = np.zeros(size)
    coeffs_exp = np.zeros(size)
    coeffs_dsl = np.zeros(size)
    coeffs_ppi = np.zeros(size)

    for i, a in enumerate(annotations):
        N = data[a]["all"].shape[0]
        subsample = np.random.choice(N, R, replace = False)
        coeffs_all[i,:,:,:] = data[a]["all"][subsample,:,:]
        coeffs_exp[i,:,:,:] = data[a]["exp"][subsample,:,:]
        coeffs_dsl[i,:,:,:] = data[a]["dsl"][subsample,:,:]
        coeffs_ppi[i,:,:,:] = data[a]["ppi"][subsample,:,:]
        num_exp[i,:] = data[a]["num_expert_samples"]

    transposed = (1, 2, 3, 0)
    reshaped = (R, 10, 5 * A)
    coeffs_all = coeffs_all.transpose(transposed).reshape(reshaped)
    coeffs_exp = coeffs_exp.transpose(transposed).reshape(reshaped)
    coeffs_dsl = coeffs_dsl.transpose(transposed).reshape(reshaped)
    coeffs_ppi = coeffs_ppi.transpose(transposed).reshape(reshaped)

    print("- computing and plotting the RMSE")
    ax.set_title(dataset)
    if norm == "raw":
        print("- using no normalisation")
        rmse_exp = compute_rmse(coeffs_all, coeffs_exp, False)
        rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl, False)
        rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi, False)
        plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi, False)
    elif norm == "per-coeff":
        print("- using per-coefficient normalisation")
        rmse_exp = compute_rmse(coeffs_all, coeffs_exp, True)
        rmse_dsl = compute_rmse(coeffs_all, coeffs_dsl, True)
        rmse_ppi = compute_rmse(coeffs_all, coeffs_ppi, True)
        plot_rmse(ax, rmse_exp, rmse_dsl, rmse_ppi, True)
    elif norm == "odds" or  norm == "logodds" or norm == "percent":
        print(f"- using ratio ({norm}) normalisation")
        rmse_dsl, rmse_ppi = compute_rmse_ratio(
            coeffs_all,
            coeffs_exp,
            coeffs_dsl,
            coeffs_ppi,
            norm,
        )
        plot_rmse_ratio(ax, rmse_dsl, rmse_ppi, norm)
    else:
        raise Exception(f"norm {norm} not supported in plot_dataset")

if __name__ == "__main__":

    args = parse_args()

    datasets = ["amazon", "misinfo", "biobias", "germeval"]
    annotations = ["bert", "deepseek", "phi4"]

    print("")
    print("Gathering the data")
    data = {d: {a: gather(d,a) for a in annotations} for d in datasets}

    print("")
    print("Plot for all datasets:")
    fig, ax = plt.subplots(figsize=(7,5))
    plot_all(ax, data, args.norm)
    plt.tight_layout()
    plt.savefig(args.plot_dir / "rmse_all.png")
    plt.savefig(args.plot_dir / "rmse_all.pdf")

    rows = 2
    cols = 2
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
    for i, dataset in enumerate(datasets):
        print("")
        print(f"Plot for dataset {dataset}")
        plot_dataset(axs[i // rows, i % cols], data[dataset], args.norm)
    plt.tight_layout()
    plt.savefig(args.plot_dir / f"rmse_datasets.png")
    plt.savefig(args.plot_dir / f"rmse_datasets.pdf")

    print("")
