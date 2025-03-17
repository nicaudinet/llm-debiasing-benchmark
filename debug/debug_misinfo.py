import os
from pathlib import Path
import pandas as pd

def missing_jobs(datapath, prefix):
    reps = set()
    for file in os.listdir(datapath):
        num = int(file[len(prefix):-len(".npz")])
        reps.add(num)
    all = set(range(1,500))
    return all - reps

err_1 = missing_jobs(
    "/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/vary-num-expert/misinfo/data/deepseek_run1",
    "data_misinfo_"
)
err_2 = missing_jobs(
    "/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/vary-num-expert/misinfo/data/deepseek",
    "data_"
)

common = sorted(list(err_1 & err_2))
print(common)


###

data = pd.read_csv("debug/account_6906541.csv")
def filter_job(job):
    return not (job.find("batch") or job.find("extern"))
data = data[~data["JobID"].str.contains("batch|extern", na=False)]

failed = data[data["State"] != "COMPLETED"]
print("")
print("Failed jobs:")
print(failed)
print(failed[failed["Elapsed"] == failed["Elapsed"].max()])

print("")
print("Elapsed time counts for failed jobs:")
for t in failed["Elapsed"].unique():
    print(f" - {t}: {len(failed[failed["Elapsed"] == t])}")
print(failed[failed["Elapsed"] == "00:00:01"])

print("")
print("Jobs in vera-r03-15:")
print(data[data["NodeList"] == "vera-r03-15"])
