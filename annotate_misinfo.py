from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
import sys
import torch

#########
# Paths #
#########

dir_path = Path(sys.argv[1])
annotated_data_path = Path(sys.argv[2])

#############
# Read data #
#############

file_path = dir_path / os.listdir(dir_path)[0]
schema = pq.read_schema(file_path)
print(schema)

data = pd.read_parquet(file_path)
print(data.head())
print(f"Total samples: {len(data)}")

sources = {s: len(data[data["source"] == s]) for s in data["source"].unique()}
sources = sorted(sources.items(), key = lambda x: x[1], reverse = True)
print("Top 10 sources")
for s in sources[:10]:
    print(f" - {s[0]}: {s[1]}")

# Filter data to only two balanced classes
thesun = data[data["source"] == "thesun"]
theguardianuk = data[data["source"] == "theguardianuk"]
n_class = min(len(thesun), len(theguardianuk))
thesun = thesun.sample(n = n_class)
theguardianuk = theguardianuk.sample(n = n_class)
data = pd.concat([thesun, theguardianuk])
print(f"Filtered data samples: {len(data)}")

#################
# Make features #
#################

original_columns = list(data.columns.values)

data["x1"] = data["content"].map(lambda x: len(x))
data["x2"] = data["content"].map(lambda x: len(x.split(" ")))
data["x3"] = data["content"].map(lambda x: sum(1 for c in x if c.isupper()))
data["x4"] = data["title"].map(lambda x: len(x))

source_map = {"thesun": 0, "theguardianuk": 1}
data["y"] = data["source"].map(lambda x: source_map[x])

#################
# Test features #
#################

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

embeddings = embed(list(data["content"]))
X = np.array(embeddings)
print(f"Embedding dimensions: {X.shape}")
Y = data["y"].to_numpy()
logreg = LogisticRegression()
logreg.fit(X, Y)
data["y_hat"] = logreg.predict(X)

#############
# Save data #
#############

data = data.drop(columns = original_columns)
print("\nFinal data:")
print(data.head())

data.to_pickle(annotated_data_path)
print(f"\nSaved data to {annotated_data_path}")
