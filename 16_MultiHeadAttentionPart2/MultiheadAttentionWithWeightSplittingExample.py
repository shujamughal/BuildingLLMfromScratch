import torch

# Set random seed for reproducibility
torch.manual_seed(0)

# Batch size = 1, number of tokens = 3, input dimension d_in = 6
batch = torch.randn(1, 3, 6)

print("Input shape:", batch.shape)
print(batch)

import torch.nn as nn

# Dimensions
d_in = 6   # input dim
d_out = 6  # output dim (for multi-head attention)
num_heads = 2
head_dim = d_out // num_heads  # = 3

# Define linear layers (random weights)
W_query = nn.Linear(d_in, d_out, bias=False)
W_key   = nn.Linear(d_in, d_out, bias=False)
W_value = nn.Linear(d_in, d_out, bias=False)

# Apply to batch
queries = W_query(batch)  # shape: [1, 3, 6]
keys    = W_key(batch)    # shape: [1, 3, 6]
values  = W_value(batch)  # shape: [1, 3, 6]

# Show shapes and values
print("Queries shape:", queries.shape)
print("Queries:\n", queries[0])
print("\nKeys:\n", keys[0])
print("\nValues:\n", values[0])

num_heads = 2
head_dim = d_out // num_heads  # 6 // 2 = 3

# Split into heads
queries = queries.view(1, 3, num_heads, head_dim).transpose(1, 2)
keys    = keys.view(1, 3, num_heads, head_dim).transpose(1, 2)
values  = values.view(1, 3, num_heads, head_dim).transpose(1, 2)
# Show shapes after splitting
print("\nQueries after splitting into heads:", queries.shape)
print("Keys after splitting into heads:", keys.shape)
print("Values after splitting into heads:", values.shape)

# Step 4: Compute attention scores per head
attn_scores = queries @ keys.transpose(2, 3)  # shape: [1, 2, 3, 3]
print("Attention Scores shape:", attn_scores.shape)
print(attn_scores)

# Create upper-triangular mask of shape [num_tokens, num_tokens]
num_tokens = attn_scores.shape[-1]
mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)
mask_bool = mask.bool()

# Apply mask to attention scores
attn_scores = attn_scores.masked_fill(mask_bool, float('-inf'))

print("Masked Attention Scores:")
print(attn_scores)
import torch.nn.functional as F

# --- Step 6: Apply Softmax and Dropout ---
head_dim = 3  # since d_out = 6 and num_heads = 2

# Normalize the attention scores into probabilities
attn_weights = torch.softmax(attn_scores / head_dim**0.5, dim=-1)

# Apply dropout to the attention weights (optional)
dropout = torch.nn.Dropout(0.0)  # No dropout for now
attn_weights = dropout(attn_weights)

print("🧮 Attention Weights:\n", attn_weights)
# --- Step 7: Compute Context Vectors ---
context_vec = attn_weights @ values  # shape: [1, 2, 3, 3]
print("📦 Context Vectors:\n", context_vec)
print("context_vec.shape:", context_vec.shape)
# --- Step 8: Merge heads ---
batch_size, num_tokens, _ = batch.shape

context_vec = context_vec.transpose(1, 2)  # [1, 3, 2, 3]
context_vec = context_vec.contiguous().view(batch_size, num_tokens, num_heads * head_dim)
print("🧩 Final Merged Context Vectors:\n", context_vec)
print("context_vec.shape:", context_vec.shape)
# --- Step 9: Final Linear Layer ---# --- Step 9: Final Linear Projection ---
out_proj = nn.Linear(d_out, d_out, bias=False)

# Apply projection
final_output = out_proj(context_vec)

print("🎯 Final Output after Projection:\n", final_output)
print("final_output.shape:", final_output.shape)

