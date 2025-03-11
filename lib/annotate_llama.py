import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from annotate_prompts import system_prompts, make_user_prompt
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from itertools import batched
import os

###########
# Options #
###########

parser = ArgumentParser()
parser.add_argument('dataset', choices = list(system_prompts.keys()))
parser.add_argument('parsed_path', type = Path)
parser.add_argument('annotation_dir', type = Path)
parser.add_argument('--num', type = int, default = 5)
parser.add_argument('--model', type = str, default = "meta-llama/Llama-3.2-3B-Instruct")
parser.add_argument('--batchsize', type = int, default = 8)
args = parser.parse_args()
print(f"Running {args.num} DeepSeek API requests")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
print("Using device:", device)

#################
# Load the data #
#################

data = pd.read_json(args.parsed_path)
data = data[:args.num]
data = data.reset_index(drop = True)

############
# Annotate #
############

data["prompt"] = data["text"].apply(lambda x: make_user_prompt(args.dataset, x))
print(data)

print(f"Loading tokenizer for {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)

print(f"Loading model for {args.model}")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map = device,
    torch_dtype = torch.float16,
)

generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 512,
    num_return_sequences = 1,
)

def make_chat(text):
    return [
        {
            "role": "system",
            "content":system_prompts[args.dataset],
        },
        {
            "role": "user",
            "content": make_user_prompt(args.dataset, text),
        },
    ]

for batch in batched(range(len(data)), args.batchsize):
    chats = [make_chat(text) for text in data["text"][batch]]
    outputs = generator(chats)
    print(outputs)
    with open("outputs.txt", "r", encoding = "utf-8") as file:
        file.write(str(outputs))

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
