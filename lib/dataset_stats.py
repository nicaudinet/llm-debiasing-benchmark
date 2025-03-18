from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import matplotlib.pyplot as plt

dataset = "germeval"
model = "microsoft/phi-4"

data = pd.read_json(f"/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/annotations/{dataset}/parsed.json")
# data = data[:10]
tokenizer = AutoTokenizer.from_pretrained(model)
data["length"] = data["text"].map(lambda x: len(tokenizer(x)["input_ids"]))
data["length"].hist(bins=30, grid=False)
plt.savefig(f"{dataset}_length_hist.pdf")
data.to_json(f"/mimer/NOBACKUP/groups/ci-nlp-alvis/dsl-use/annotations/{dataset}/tokenized.json")

print(data)
print("Mean length:", data["length"].mean())
print("Std length:", data["length"].std())
print("Max length:", data["length"].max())
print("Number of reviews over 500 tokens:", len(data[500 < data["length"]]))
print("Number of reviews over 1000 tokens:", len(data[1000 < data["length"]]))
