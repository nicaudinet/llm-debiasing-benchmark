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

train = pd.read_csv(data_path / "train.csv")
test = pd.read_csv(data_path / "test.csv")
data = pd.concat([train, test], axis = 0, ignore_index = True)
print(data)

#######################
# Balance the dataset #
#######################

other = data[data["binary"] == "OTHER"]
offense = data[data["binary"] == "OFFENSE"]
samples_per_class = min(len(other), len(offense))
print(samples_per_class)
other = other.sample(samples_per_class)
offense = offense.sample(samples_per_class)
data = pd.concat([other, offense], axis = 0, ignore_index = True)
print(data)

####################
# Extract features #
####################

data["x1"] = data["text"].map(lambda x: sum(1 for c in x if c == '@'))
data["x2"] = data["text"].map(lambda x: len(x))
data["x3"] = data["text"].map(lambda x: len(x.split(" ")))
data["x4"] = data["text"].map(lambda x: sum(1 for c in x if c.isupper()))

data["y"] = data["binary"].map(lambda x: 0 if x == "OTHER" else 1)

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

embeddings = embed(list(data["text"]))
X = np.array(embeddings)
print(f"Embedding dimensions: {X.shape}")
Y = data["y"].to_numpy()
logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X, Y)
data["y_hat"] = logreg.predict(X)

#############
# Save data #
#############

print("\nFinal data:")
print(data.head())

data.to_pickle(annotated_data_path)
print(f"\nSaved data to {annotated_data_path}")
