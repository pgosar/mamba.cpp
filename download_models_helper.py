from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")

input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]
out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
