import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Setup paths
base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./checkpoints/dpo" # Path to your finished training folder

# 2. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 3. Load the Base Model (Optimized for Mac/MPS)
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16, # Use float16 for faster testing
    device_map="mps"           # Use your Mac's GPU
)

# 4. Load the AegisRAG Adapter
print("Applying AegisRAG DPO adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Updated Step 5
messages = [
    {"role": "system", "content": "You are a helpful assistant. Use only the provided context."},
    {"role": "user", "content": "Context: The company reported strong sales in 2023. \n\nQuestion: What was the revenue growth in 2024?"}
]

# Apply the Qwen-2.5 template
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("mps")

outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False) # do_sample=False for consistent results


print("\n--- Model Output ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))