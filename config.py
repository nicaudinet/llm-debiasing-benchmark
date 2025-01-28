from pathlib import Path

# Paths
original_reviews_path = Path("/Users/audinet/Datasets/amazon_reviews/original_reviews.txt")
annotated_reviews_path = Path("resources/amazon_annotated.pkl")

# Data annotation
num_samples = 10
model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
revision = "714eb0f"
