from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

import inspect

prompt = "Hey how are you doing?"

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")

# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
# model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
# model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")

tokens = tokenizer(prompt, return_tensors="pt")
input_ids = tokens.input_ids
out = model.generate(input_ids, max_new_tokens=15)

print(input_ids)
print(input_ids.__class__.__name__)
print(input_ids.dtype)
print(input_ids.shape)
print(hasattr(model, "_decoding_cache"))

print("\n~~~~~~~~~~~~~\n")

print(tokenizer.batch_decode(out))