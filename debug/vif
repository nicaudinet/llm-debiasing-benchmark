import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(dataset):
    data = pd.read_json(f"/Users/audinet/Projects/work/dsl-use/results/annotations/{dataset}/parsed.json")

    data = data[["x1", "x2", "x3", "x4"]]
    data.insert(0,'Intercept', 1)

    # Compute VIF for each independent variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    # Display results
    print("")
    print(dataset)
    for i in vif_data.index:
        print(f" - {vif_data["Variable"][i]}: {vif_data["VIF"][i]}")

if __name__ == "__main__":

    datasets = ["amazon", "misinfo", "biobias", "germeval"]
    for dataset in datasets:
        vif(dataset)
