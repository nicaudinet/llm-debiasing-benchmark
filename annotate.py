from pathlib import Path
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

datapath = Path("/Users/audinet/Datasets/amazon_reviews/original_reviews.txt")
with open(datapath, "r") as file:
    reviews = file.read().splitlines()

data = {k: [] for k in ["topic", "sentiment", "filename", "text"]}
for review in reviews:
    words = review.split(" ")
    data["topic"].append(words[0])
    data["sentiment"].append(1 if words[1] == "pos" else 0)
    data["filename"].append(words[2])
    # Truncate reviews to 512 characters so that review fits into distilbert
    data["text"].append(" ".join(words[3:])[:512])

data = pd.DataFrame(data)
# data = data[:10] # REMOVE ME
print(data)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision = "714eb0f",
)

sentiments = sentiment_pipeline(data["text"].tolist())
sentiments = [1 if d["label"] == "POSITIVE" else 0 for d in sentiments]
data["llm_annotations"] = sentiments
print(data)

data.to_pickle("amazon_reviews_annotated.pkl")

accuracy = accuracy_score(data["sentiment"], data["llm_annotations"])
print(f"\nLLM Accuracy (all samples): {accuracy:0.3f}")

print("\nAccuracies by topic:")
for topic in pd.unique(data["topic"]):
    d = data[data["topic"] == topic]
    accuracy = accuracy_score(d["sentiment"], d["llm_annotations"])
    print(f" - {topic}    \t{accuracy:0.3f}")
