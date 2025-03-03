from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import pandas as pd
import sys
import torch

#########
# Paths #
#########

data_path = Path(sys.argv[1])
annotated_data_path = Path(sys.argv[2])

#######################
# Balance the dataset #
#######################

data = pd.read_json(data_path)

labels = data[["profession", "gender"]]
data["label"] = labels.apply(lambda x: "_".join(x.astype(str)), axis = 1)
num_samples = 10000
num_labels = len(data["label"].unique())
samples_per_label = num_samples // num_labels
samples = []
for label in data["label"].unique():
    rows = data[data["label"] == label]
    sampled_rows = rows.sample(min(len(rows), samples_per_label))
    samples.append(sampled_rows)
samples = pd.concat(samples, axis = 0, ignore_index = True)
remaining = data[~data["hard_text"].isin(samples["hard_text"])]
remaining = remaining.sample(num_samples - len(samples))
data = pd.concat([samples, remaining], axis = 0, ignore_index = True)
data = data.drop("label", axis = 1)

#####################
# Check the samples #
#####################

print(samples)

print("Gender balance:")
for gender in samples["gender"].unique():
    print(f" - {gender}: {len(samples[samples["gender"] == gender])}")

print("Profession balance:")
for profession in samples["profession"].unique():
    print(f" - {profession}: {len(samples[samples["profession"] == profession])}")

####################
# Extract features #
####################

data["x1"] = data["profession"]
data["x2"] = data["hard_text"].map(lambda x: len(x))
data["x3"] = data["hard_text"].map(lambda x: len(x.split(" ")))
data["x4"] = data["hard_text"].map(lambda x: sum(1 for c in x if c.isupper()))

data["y"] = data["gender"]

####################
# Measure features #
####################

X = data[["x1","x2","x3","x4"]].to_numpy()
Y = data["y"].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")

############
# Annotate #
############

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device {device}")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

def embed(texts, batch_size = 32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = 512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy() # Use CLS token
        embeddings.extend(batch_embeddings)
    return embeddings

embeddings = embed(list(data["hard_text"]))
X = np.array(embeddings)
print(f"Embedding dimensions: {X.shape}")
Y = data["y"].to_numpy()
logreg = LogisticRegression()
logreg.fit(X, Y)
data["y_hat"] = logreg.predict(X)

#############
# Save data #
#############

print("\nFinal data:")
print(data.head())

data.to_pickle(annotated_data_path)
print(f"\nSaved data to {annotated_data_path}")
