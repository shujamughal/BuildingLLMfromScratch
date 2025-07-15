import torch
import torch.nn as nn

# Dummy sentence embeddings for 6 tokens (e.g., "Your journey starts with one step")
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your
    [0.55, 0.87, 0.66],  # Journey
    [0.57, 0.85, 0.64],  # Starts
    [0.22, 0.58, 0.33],  # With
    [0.77, 0.25, 0.10],  # One
    [0.05, 0.80, 0.55],  # Step
])

# Set seed for reproducibility
torch.manual_seed(123)

# Define input/output dimensions
d_in = inputs.shape[-1]
d_out = 2  # output embedding size

# Create trainable weight layers
W_query = nn.Linear(d_in, d_out, bias=False)
W_key   = nn.Linear(d_in, d_out, bias=False)

# Compute queries and keys
queries = W_query(inputs)  # shape: [6, 2]
keys    = W_key(inputs)    # shape: [6, 2]

# Compute dot-product attention scores
attn_scores = queries @ keys.T  # shape: [6, 6]

# Scale scores and apply softmax
attn_weights = torch.softmax(attn_scores / d_out**0.5, dim=-1)

# Show result (before masking)
print("Attention Weights (before masking):")
print(attn_weights)

context_len = attn_scores.shape[0]  # → 6
mask = torch.tril(torch.ones(context_len, context_len))
print(mask)
# Apply the mask to attention weights
masked_weights = attn_weights * mask
print(masked_weights)
# Normalize masked weights
row_sums = masked_weights.sum(dim=-1, keepdim=True)
masked_weights_norm = masked_weights / row_sums
print(masked_weights_norm)

context_length = attn_scores.shape[0]  # 6 in our case

# 1. Create a mask with 1s above the diagonal (future positions)
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

# 2. Replace those 1s with -inf (so they become zero in softmax)
masked_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)

print(masked_scores)

attn_weights = torch.softmax(masked_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)      # 50% dropout rate
example = torch.ones(6, 6)           # Simulated weight matrix
print(dropout(example))

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)

masked_attn_weights = torch.softmax(masked_scores / keys.shape[-1]**0.5, dim=1)
dropped_attn_weights = dropout(masked_attn_weights)

print(dropped_attn_weights)

import torch

torch.manual_seed(123)

inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your
    [0.55, 0.87, 0.66],  # Journey
    [0.57, 0.85, 0.64],  # Starts
    [0.22, 0.58, 0.33],  # With
    [0.77, 0.25, 0.10],  # One
    [0.05, 0.80, 0.55],  # Step
])

batch = torch.stack((inputs, inputs), dim=0)
print("Batch shape:", batch.shape)

import torch.nn as nn
torch.manual_seed(42)  # Set the seed here

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # 📌 Create upper-triangular mask (zeros below and on diag)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Compute raw attention scores: [batch, tokens, tokens]
        attn_scores = queries @ keys.transpose(1, 2)

        # Apply causal mask with -inf above diagonal
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # Softmax + dropout
        attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute final context vectors
        context_vec = attn_weights @ values
        return context_vec
d_in = 3
d_out = 2
context_length = batch.shape[1]

ca = CausalAttention(d_in, d_out, context_length, dropout=0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:", context_vecs)