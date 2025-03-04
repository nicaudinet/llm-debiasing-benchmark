from openai import OpenAI
from pathlib import Path
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os

# TODO: sleep a bit between API calls (avoid throttling)
#   - try/except error handling
#   - look into batching

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
    'annotation_path',
    type = Path,
    help='Path to the annotations'
)

parser.add_argument(
    '--num',
    '-n',
    type = int,
    default=5,
    help='Number of reviews to process (default: 5)'
)

args = parser.parse_args()
print(f"Running {args.num} DeepSeek API requests")

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

#####################
# DeepSeek API call #
#####################

with open(".deepseek_api_key", "r") as file:
    deepseek_api_key = file.read().strip()

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
    return prompt, response.choices[0].message.content

prompt_path = args.annotation_path / Path("prompts")
os.makedirs(prompt_path, exist_ok = True)

response_path = args.annotation_path / Path("responses")
os.makedirs(response_path, exist_ok = True)

responses = {}

with ThreadPoolExecutor(max_workers=args.num) as executor:

    futures = {}

    for index, prompt in enumerate(data["prompt"]):
        future = executor.submit(call_deepseek, prompt)
        futures[future] = index

    for future in as_completed(futures):

        index = futures[future]

        try:

            prompt, response = future.result()
            responses[index] = response

            prompt_file = prompt_path / Path(f"prompt_{index:04}.txt")
            with open(prompt_file, "w", encoding="utf-8") as file:
                file.write(prompt)

            response_file = response_path / Path(f"response_{index:04}.txt")
            with open(response_file, "w", encoding="utf-8") as file:
                file.write(response)

        except Exception as exception:
            print(f"Prompt {index:04} failed with exception: {exception}") 

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

invalid_responses = {}
annotations = {}
for index, response in responses.items():
    try:
        annotations[index] = parse_annotation(response)
    except ValueError as error:
        invalid_responses[index] = response
    except Exception as exception:
        print(f"Some exception occurred when parsing responses: {exception}")

print(f"Responses that failed to parse: {len(invalid_responses)}")
print(annotations)

data = data.take(list(annotations.keys()))
data["annotations"] = annotations.values()
print(data)

final_path = args.annotation_path / Path("annotations.pkl")
data.to_pickle(final_path)
print(f"\nSaved data to {final_path}")
