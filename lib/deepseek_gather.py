from openai import OpenAI
from pathlib import Path
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
from itertools import batched
import time

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'in_dir',
        type = Path,
        help = 'Path to the annotation folder'
    )

    parser.add_argument(
        'out_path',
        type = Path,
        help='Path to save the dataframe'
    )

    return parser.parse_args()

def load_data(path):
    """
    Load the Amazon Reviews dataset as a pandas DataFrame
    """

    with open(path, "r") as file:
        reviews = file.read().splitlines()

    data = {k: [] for k in ["topic", "sentiment", "filename", "text"]}
    for review in reviews:
        words = review.split(" ")
        data["topic"].append(words[0])
        data["sentiment"].append(1 if words[1] == "pos" else 0)
        data["filename"].append(words[2])
        data["text"].append(" ".join(words[3:]))

    data = pd.DataFrame(data)
    return data

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

def call_deepseek(client, prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a perfect sentiment classification system"},
            {"role": "user", "content": prompt },
        ],
        stream = False
    )
    return prompt, response.choices[0].message.content

def save_prompt(annotation_path, prompt, index):
    prompt_path = annotation_path / Path("prompts")
    os.makedirs(prompt_path, exist_ok = True)
    prompt_file = prompt_path / Path(f"prompt_{index:04}.txt")
    with open(prompt_file, "w", encoding="utf-8") as file:
        file.write(prompt)

def save_response(annotation_path, response, index):
    response_path = annotation_path / Path("responses")
    os.makedirs(response_path, exist_ok = True)
    response_file = response_path / Path(f"response_{index:04}.txt")
    with open(response_file, "w", encoding="utf-8") as file:
        file.write(response)

def annotate(client, prompts, annotation_path):

    responses = {}

    with ThreadPoolExecutor(max_workers = len(prompts)) as executor:

        futures = {}

        for index, prompt in prompts.items():
            future = executor.submit(call_deepseek, client, prompt)
            futures[future] = index

        for future in as_completed(futures):
            index = futures[future]

            try:
                prompt, response = future.result()
                responses[index] = response
                save_prompt(annotation_path, prompt, index)
                save_response(annotation_path, response, index)

            except Exception as exception:
                print(f"Prompt {index:04} failed with exception: {exception}") 

    return responses

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



if __name__ == "__main__":

    args = parse_args()

    responses = {}
    response_path = args.in_dir / Path("responses")
    for file in os.listdir(response_path):
        index = int(file[len("response_"):-len(".txt")])
        with open(response_path / file, "r", encoding = "utf-8") as f:
            responses[index] = f.read().strip()

    prompts = {}
    for index in responses.keys():
        try:
            prompt_file = args.in_dir / Path(f"prompts/prompt_{index:04}.txt")
            with open(prompt_file, "r", encoding = "utf-8") as f:
                prompts[index] = f.read().strip()
        except Exception as e:
            print(f"Couldn't find prompt file for index {index}: {e}")

    invalid_responses = {}
    annotations = {}
    for index, response in responses.items():
        try:
            annotations[index] = parse_annotation(response)
        except ValueError as error:
            print(f"Value error at index {index}: {error}")
            print("- Response:", response)
            invalid_responses[index] = response
        except Exception as exception:
            print(f"Some exception occurred when parsing responses: {exception}")

    print(len(invalid_responses))

    data = pd.DataFrame.from_dict(
        annotations,
        orient = "index",
        columns = ["annotation"]
    )
    data.reset_index(inplace = True)
    data["prompt"] = data["index"].map(prompts)
    data["response"] = data["index"].map(responses)
    print(data)

    data.to_pickle(args.out_path)
    print(f"\nSaved data to {args.out_path}")
