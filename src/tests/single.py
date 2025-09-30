from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
HF_TOKEN = os.getenv("HF_TOKEN")


model_id = "neulab/Pangea-7B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=HF_TOKEN,
    trust_remote_code=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,   # or float32 if no GPU
    device_map="auto",
    use_auth_token=HF_TOKEN,
    trust_remote_code=True
)

# Simple inference
prompt = "Translate the following Arabic sentence to English: مرحبا كيف حالك؟"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.1
    )

print("=== Model output ===")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
