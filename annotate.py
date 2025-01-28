import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score
import re

import config

###########################
# Parse data to DataFrame #
###########################

with open(config.original_reviews_path, "r") as file:
    reviews = file.read().splitlines()

data = {k: [] for k in ["topic", "sentiment", "filename", "text"]}
for review in reviews:
    words = review.split(" ")
    data["topic"].append(words[0])
    data["sentiment"].append(1 if words[1] == "pos" else 0)
    data["filename"].append(words[2])
    data["text"].append(" ".join(words[3:]))

data = pd.DataFrame(data)
data = data[:config.num_samples]
original_labels = list(data.columns.values)

####################
# Extract features #
####################

# Encode topic as a feature
topics = pd.unique(data["topic"])
topicNum = dict(zip(topics, range(len(topics) + 1)))
data["x1"] = data["topic"].map(lambda x: topicNum[x])

# Encode review length as a feature
data["x2"] = data["text"].map(lambda x: len(x))

# Encode "number of repetitions of the word i" as a feature
def repetitions_i(text):
    return sum(1 for _ in re.finditer(r"\b%s\b" % re.escape("i"), text))
data["x3"] = data["text"].map(lambda x: repetitions_i(x))

######################
# Annotate with BERT #
######################

# Truncate reviews to 512 characters so that review fits into distilbert
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model = config.model,
    revision = config.revision,
)
text_truncated = data["text"].map(lambda x: x[:512]).tolist()
sentiments = sentiment_pipeline(text_truncated)
sentiments = [1 if d["label"] == "POSITIVE" else 0 for d in sentiments]

# Encode y and y_hat
data["y"] = data["sentiment"]
data["y_hat"] = sentiments

###################
# LLM Accuracices #
###################

accuracy = accuracy_score(data["y"], data["y_hat"])
print(f"\nLLM Accuracy (all samples): {accuracy:0.3f}")

print("\nAccuracies by topic:")
for topic in pd.unique(data["topic"]):
    d = data[data["topic"] == topic]
    accuracy = accuracy_score(d["y"], d["y_hat"])
    print(f" - {topic}    \t{accuracy:0.3f}")

##############
# Save data # 
##############

data = data.drop(columns = original_labels)
print("\nFinal data:")
print(data.head())

data.to_pickle(config.annotated_reviews_path)
print(f"\nSaved data to {config.annotated_reviews_path}")
