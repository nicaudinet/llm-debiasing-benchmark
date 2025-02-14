from datasets import load_dataset, Features, Value
import pandas as pd
import os
from pathlib import Path
import pyarrow.parquet as pq

dir_path = Path("/Users/audinet/Datasets/misinfo-general/data/")
# base_path = Path("/Users/audinet/.cache/huggingface/hub/datasets--ioverho--misinfo-general/blobs")
#
# for filename in os.listdir(dir_path):
#     file_path = dir_path / Path(filename)
#     original_target = file_path.readlink()
#     blob_hash = original_target.parts[-1]
#     new_path = base_path / blob_hash
#     os.unlink(file_path)
#     os.symlink(new_path, file_path)

file_path = dir_path / os.listdir(dir_path)[0]
schema = pq.read_schema(file_path)
print(schema)

data = pd.read_parquet(file_path)
print(data.head())

features = Features({
    'source': Value('string'),
    'title': Value('string'),
    'content': Value('string'),
    'author': Value('string'),
    'domain': Value('string'),
    'raw_url': Value('string'),
    'publication_date': Value('timestamp[ns]'),
    'article_id': Value('string')
})

data = load_dataset(
    "parquet", 
    data_files="/Users/audinet/Datasets/misinfo-general/data/*.parquet",
    features=features,
    split="train",
)

# data = load_dataset(
#     "parquet",
#     data_files = "/Users/audinet/Datasets/misinfo-general/data/*.parquet"
# )
# print(data)
