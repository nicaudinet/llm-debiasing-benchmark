from openai import OpenAI
from pathlib import Path
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
from itertools import batched

from annotate_prompts import dataset_labels, system_prompts, make_user_prompt

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

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices = list(system_prompts.keys()))
    parser.add_argument('parsed_path', type = Path)
    parser.add_argument('annotation_dir', type = Path)
    parser.add_argument('--num', type = int, default = 5)
    parser.add_argument('--num_examples', type = int, default = 0)
    args = parser.parse_args()

    data = pd.read_json(args.parsed_path)
    data = data[:args.num]
    data = data.reset_index(drop = True)

    examples = data.sample(min(args.num, args.num_examples))
    data = data.loc[~data.index.isin(examples.index)]
    labels = examples["y"].apply(lambda i: dataset_labels[args.dataset][i])
    examples = list(zip(examples["text"], labels))

    data["prompt"] = data["text"].apply(lambda x: make_user_prompt(args.dataset, x, examples))

    with open(".deepseek_api_key", "r") as file:
        deepseek_api_key = file.read().strip()

    client = OpenAI(
        api_key = deepseek_api_key,
        base_url = "https://api.deepseek.com"
    )

    print(f"Running {len(data)} DeepSeek API requests")
    print(data)
    system_prompt = system_prompts[args.dataset]
    for selection in batched(data.index, n = 200):
        selection = list(selection)
        print(f"Running batch {selection[0]:04} - {selection[-1]:04}")
        user_prompts = {i: data["prompt"][i] for i in selection}
        annotate(client, system_prompt, user_prompts, args.annotation_dir)
