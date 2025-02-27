from openai import OpenAI
from pathlib import Path
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor
import argparse

##############
# Parameters #
##############

parser = argparse.ArgumentParser(
    description = 'Process original and annotated paths with optional review count.'
)

parser.add_argument(
    'original_path',
    type = Path,
    help = 'Path to the original file'
)

parser.add_argument(
    'annotated_path',
    type = Path,
    help='Path to the annotated file'
)

parser.add_argument(
    '--num',
    '-n',
    type = int,
    default=5,
    help='Number of reviews to process (default: 5)'
)

args = parser.parse_args()

###########################
# Parse data to DataFrame #
###########################

with open(args.original_path, "r") as file:
    reviews = file.read().splitlines()

data = {k: [] for k in ["topic", "sentiment", "filename", "text"]}
for review in reviews:
    words = review.split(" ")
    data["topic"].append(words[0])
    data["sentiment"].append(1 if words[1] == "pos" else 0)
    data["filename"].append(words[2])
    data["text"].append(" ".join(words[3:]))

data = pd.DataFrame(data)
data = data[:args.num]
print(data)

##########
# Prompt #
##########

def make_prompt(text):
    return f"""
Classify the following text as either POSITIVE or NEGATIVE. Give no other
explanation for your classification, only output the label.

Here are a few examples:

I love you

CLASSIFICATION: POSITIVE

I hate you

CLASSIFICATION: NEGATIVE

Here's the text to classify:

{text}

CLASSIFICATION: 
"""

data["prompt"] = data["text"].apply(make_prompt)
print(data["prompt"][0])

#####################
# DeepSeek API call #
#####################

with open(".deepseek_api_key", "r") as file:
    deepseek_api_key = file.read().strip()
print(deepseek_api_key)

client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

def call_deepseek(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a perfect sentiment classification system"},
            {"role": "user", "content": prompt },
        ],
        stream = False
    )
    return response.choices[0].message.content

with ThreadPoolExecutor(max_workers=args.num) as executor:
    responses = list(executor.map(call_deepseek, data["prompt"]))

print(responses)

######################
# Parse LLM Response #
######################

def parse_annotation(text):
    pos_count = text.count("POSITIVE")
    neg_count = text.count("NEGATIVE")
    if pos_count > 0 and neg_count > 0:
        raise ValueError("Response contains both labels")
    elif pos_count > 1:
        raise ValueError("Response contains more than one POSITIVE label")
    elif neg_count > 1:
        raise ValueError("Response contains more than one NEGATIVE label")
    elif pos_count == 1:
        return 1
    elif neg_count == 1:
        return 0
    else:
        raise ValueError("Response didn't contain any label")

data["annotation"] = list(map(parse_annotation, responses))
print(data)

data.to_pickle(args.annotated_path)
print(f"\nSaved data to {args.annotated_path}")
