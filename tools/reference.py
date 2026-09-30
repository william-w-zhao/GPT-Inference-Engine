import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

# "the cat sat on the"
ids = torch.tensor([[464, 3797, 3332, 319, 262]])

with torch.no_grad():
    logits = model(ids).logits[0, -1]

top = torch.topk(logits, 5)
for i, v in zip(top.indices.tolist(), top.values.tolist()):
    print(f"{i:6d}  {v:.4f}  {tokenizer.decode([i])!r}")