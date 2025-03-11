import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
print("Using device:", device)

model_id = "meta-llama/Llama-3.2-3B-Instruct"

print(f"Loading tokenizer for {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Loading model for {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = device,
    torch_dtype = torch.float16,
)

pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 512,
)

user_input = "why is the sea red?"
full_response = pipe(user_input)[0]["generated_text"]
model_response = full_response[len(user_input):].strip()
print("\nLlama:", model_response)
