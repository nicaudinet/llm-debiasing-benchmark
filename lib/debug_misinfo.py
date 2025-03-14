import os
from pathlib import Path

datapath = "/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/experiments/vary-num-expert/misinfo/data/deepseek"
reps = set()
for file in os.listdir(datapath):
    num = int(file[len("data_misinfo_"):-len(".npz")])
    reps.add(num)
all = set(range(500))
err = all - reps
print(err)
print(len(err))
print(len(reps))
