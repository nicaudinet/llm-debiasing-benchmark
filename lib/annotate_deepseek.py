from openai import OpenAI
from pathlib import Path
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
from itertools import batched

###########
# Prompts #
###########

def amazon_prompt(text):
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

def misinfo_prompt(text):
    return f"""
Classify the following article as either:
- THESUN if it is likely to have been published in the British tabloid newspaper
  The Sun
- THEGUARDIAN if it is likely to have been published in the British daily
  newspaper The Guardian

Give no other explanation for your classification, only output the label.

Here's an example of the formatting I would like you use, where < ARTICLE_TEXT >
is a stand-in for the article text:

< ARTICLE_TEXT >

CLASSIFICATION: THESUN

< ARTICLE_TEXT >

CLASSIFICATION: THESUN

Here's the article I would like you to classify:

{text}

CLASSIFICATION: 
"""

def biobias_prompt(text):
    return f"""
Classify the following textual biographies as either:
- MALE if the subject is likely to be male
- FEMALE if the subject is likely to be female

Give no other explanation for your classification, only output the label.

Here are a couple example of the formatting I would like you use, where < BIOGRAPHY >
is a stand-in for the textual biography:

< BIOGRAPHY >

CLASSIFICATION: MALE

< BIOGRAPHY >

CLASSIFICATION: FEMALE

Here's the textual biography I would like you to classify:

{text}

CLASSIFICATION: 
"""

def germeval_prompt(text):
    return f"""
Classify the following German tweets as either:
- OFFENSIVE if the tweet is likely to contain an offense or be offensive
- OTHER if the tweet is _not_ likely to contain an offense of be offensive

Give no other explanation for your classification, only output the label.

Here are a couple example of the formatting I would like you use, where < TWEET >
is a stand-in for the text of the tweet:

< TWEET >

CLASSIFICATION: OFFENSIVE

< BIOGRAPHY >

CLASSIFICATION: OTHER

Here's the German tweet I would like you to classify:

{text}

CLASSIFICATION: 
"""

def make_user_prompt(dataset, text):
    if dataset == "amazon":
        return amazon_prompt(text)
    elif dataset == "misinfo":
        return misinfo_prompt(text)
    elif dataset == "biobias":
        return biobias_prompt(text)
    elif dataset == "germeval":
        return germeval_prompt(text)
    else:
        raise ValueError(f"'{dataset}' is not one of the known datasets.")

############
# API Call #
############

def call_deepseek(client, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt },
        ],
        stream = False
    )
    return user_prompt, response.choices[0].message.content

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

def annotate(client, system_prompt, user_prompts, annotation_path):

    responses = {}

    with ThreadPoolExecutor(max_workers = len(user_prompts)) as executor:

        futures = {}

        for index, user_prompt in user_prompts.items():
            future = executor.submit(call_deepseek, client, system_prompt, user_prompt)
            futures[future] = index

        for future in as_completed(futures):
            index = futures[future]

            try:
                user_prompt, response = future.result()
                responses[index] = response
                save_prompt(annotation_path, user_prompt, index)
                save_response(annotation_path, response, index)

            except Exception as exception:
                print(f"Prompt {index:04} failed with exception: {exception}") 

    return responses

########
# Main #
########

if __name__ == "__main__":

    system_prompts = {
        "amazon": "You are a perfect sentiment classification system",
        "misinfo": "You are a perfect newspaper article classification system",
        "biobias": "You are a perfect biography classification system",
        "germeval": "You are a perfect German tweet classification system",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices = list(system_prompts.keys()))
    parser.add_argument('parsed_path', type = Path)
    parser.add_argument('annotation_dir', type = Path)
    parser.add_argument('--num', type = int, default = 5)
    args = parser.parse_args()
    print(f"Running {args.num} DeepSeek API requests")

    data = pd.read_json(args.parsed_path)
    data = data[:args.num]
    data = data.reset_index(drop = True)
    data["prompt"] = data["text"].apply(lambda x: make_user_prompt(args.dataset, x))
    print(data)

    with open(".deepseek_api_key", "r") as file:
        deepseek_api_key = file.read().strip()

    client = OpenAI(
        api_key = deepseek_api_key,
        base_url = "https://api.deepseek.com"
    )

    system_prompt = system_prompts[args.dataset]
    for selection in batched(range(len(data)), n = 200):
        selection = list(selection)
        print(f"Running batch {selection[0]:04} - {selection[-1]:04}")
        user_prompts = {i: data["prompt"][i] for i in selection}
        annotate(client, system_prompt, user_prompts, args.annotation_dir)
