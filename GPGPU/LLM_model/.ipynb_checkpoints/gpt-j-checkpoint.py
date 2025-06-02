import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# Track total start time
overall_start = time.time()

# Step 1: Load model and tokenizer
start = time.time()
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
end = time.time()
print(f"[Load model & tokenizer] {end - start:.3f} seconds")

# Step 2: Tokenize
prompt = "Explain how gravity works."
start = time.time()
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
end = time.time()
print(f"[Tokenization & move to GPU] {end - start:.3f} seconds")

# Step 3: Inference
start = time.time()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )
torch.cuda.synchronize()  # ensure timing is accurate
end = time.time()
print(f"[Inference] {end - start:.3f} seconds")

# Step 4: Decode
start = time.time()
result = tokenizer.decode(output[0], skip_special_tokens=True)
end = time.time()
print(f"[Decode] {end - start:.3f} seconds")

# Print result
print("\n[Output]:")
print(result)

# Total runtime
overall_end = time.time()
print(f"\n[Total time] {overall_end - overall_start:.3f} seconds")
