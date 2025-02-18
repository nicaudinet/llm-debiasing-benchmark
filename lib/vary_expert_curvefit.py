import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
from scipy.interpolate import interp1d

datadir = Path(sys.argv[1])
plotdir = Path(sys.argv[2])

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
X = data["num_expert_samples"]

# Compute RMSE with all
rmse_exp = np.sqrt(np.mean((coeffs_all - coeffs_exp) ** 2, axis=(0,2)))
rmse_dsl = np.sqrt(np.mean((coeffs_all - coeffs_dsl) ** 2, axis=(0,2)))

assert rmse_exp.shape[0] == coeffs_exp.shape[1]
assert rmse_dsl.shape[0] == coeffs_dsl.shape[1]

print(X.shape)
print(rmse_exp.shape)
print(rmse_dsl.shape)
print(coeffs_all.shape)
print(coeffs_exp.shape)
print(coeffs_dsl.shape)

inverse_spline = interp1d(rmse_exp, X, kind = "cubic", fill_value = "extrapolate")
effective_size = inverse_spline(rmse_dsl)
effective_diff = effective_size - rmse_exp

print(effective_diff)

plt.plot(X, rmse_exp, "-o", label="exp")
plt.plot(X, rmse_dsl, "-o", label="dsl")
plt.plot(effective_size, rmse_dsl, "-o", label="fit")
plt.xscale('log')
plt.legend()
plt.show()
