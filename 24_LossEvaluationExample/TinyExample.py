import torch

from FullGPTExampleWithTextGeneration import GPTModel, token_ids_to_text

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # shorter for speed
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

inputs = torch.tensor([
    [16833,  3626,  6100],   # "every effort moves"
    [   40,  1107,   588]    # "I really like"
])

targets = torch.tensor([
    [ 3626,  6100,   345],   # "effort moves you"
    [ 1107,   588, 11311]    # "really like chocolate"
])
with torch.no_grad():  # no gradient needed for evaluation
    logits = model(inputs)  
    print(logits.shape)  # Shape: [batch=2, tokens=3, vocab=50257]
    # Shape: [batch=2, tokens=3, vocab=50257]
probas = torch.softmax(logits, dim=-1)
print(probas.shape)  # [2, 3, 50257]

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
import tiktoken

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

print("Targets batch 1:", token_ids_to_text(targets[0], tokenizer))
print("Outputs batch 1:", token_ids_to_text(token_ids[0].flatten(), tokenizer))


target_probas_1 = probas[0, [0, 1, 2], targets[0]]
target_probas_2 = probas[1, [0, 1, 2], targets[1]]

print("Text 1 target probs:", target_probas_1)
print("Text 2 target probs:", target_probas_2)

# Combine all target probabilities into one tensor
all_target_probas = torch.cat((target_probas_1, target_probas_2))

# Step 1: log probabilities
log_probas = torch.log(all_target_probas)

# Step 2: average log probability
avg_log_probas = torch.mean(log_probas)

# Step 3: negate to get loss
neg_avg_log_probas = -avg_log_probas
print("Manual Loss:", neg_avg_log_probas)  
# Example output: tensor(10.7940)

logits_flat = logits.flatten(0, 1)   # Shape: [6, 50257]
targets_flat = targets.flatten()     # Shape: [6]

import torch.nn.functional as F
loss = F.cross_entropy(logits_flat, targets_flat)
print("PyTorch Loss:", loss)  # Same as manual calculation
perplexity = torch.exp(loss)
print("Perplexity:", perplexity)
