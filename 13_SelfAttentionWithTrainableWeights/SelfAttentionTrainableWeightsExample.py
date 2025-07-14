import torch

# Six tokens, each represented by a 3-dimensional embedding
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # x₁ = "Your"
    [0.55, 0.87, 0.66],  # x₂ = "Journey" ← Our Query
    [0.57, 0.85, 0.64],  # x₃ = "Starts"
    [0.22, 0.58, 0.33],  # x₄ = "With"
    [0.77, 0.25, 0.10],  # x₅ = "One"
    [0.05, 0.80, 0.55],  # x₆ = "Step"
])

d_in = inputs.shape[1]   # input embedding size = 3
d_out = 2                # output dimension for Q, K, V projections

# Seed for reproducibility
torch.manual_seed(123)

# Random (non-trainable for this example) weight matrices
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

x_2 = inputs[1]  # "Journey"

query_2 = x_2 @ W_query
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value

print("Query for x₂:", query_2)
print("Key for x₂:", key_2)
print("Value for x₂:", value_2)

keys   = inputs @ W_key     # Shape: [6, 2]
values = inputs @ W_value   # Shape: [6, 2]

# Compute attention scores
# Compute dot product of query_2 with each key vector
attn_scores_2 = torch.empty(inputs.shape[0])
for i, k_i in enumerate(keys):
    attn_scores_2[i] = torch.dot(query_2, k_i)

print("Attention Scores for x₂ (journey):")
print(attn_scores_2)

# using matrix multiplication
attn_scores_2_ = query_2 @ keys.T
print(attn_scores_2_) # same as above


d_k = keys.shape[-1]  # this is 2
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Attention Weights:", attn_weights_2)

context_vec_2 = attn_weights_2 @ values  # shape: [2]
print("Context Vector for 'journey':", context_vec_2)

# Step 1: Compute All Attention Scores (scaled dot product)
queries = inputs @ W_query  # Shape: [6, 2]
attn_scores = queries @ keys.T
print("Raw attention scores:", attn_scores.shape)
d_k = keys.shape[-1]
scaled_attn_scores = attn_scores / d_k**0.5
print("Scaled attention scores:", scaled_attn_scores)
attn_weights = torch.softmax(scaled_attn_scores, dim=-1)
print("Attention weights (all tokens):", attn_weights.shape)
all_context_vectors = attn_weights @ values
print("All context vectors:", all_context_vectors.shape)
print("Context vector for all tokens:", all_context_vectors)

import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # Step 1: Linear projections
        queries = x @ self.W_query  # [seq_len, d_out]
        keys    = x @ self.W_key    # [seq_len, d_out]
        values  = x @ self.W_value  # [seq_len, d_out]

        # Step 2: Scaled Dot-Product Attention
        attn_scores = queries @ keys.T  # [seq_len, seq_len]
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Step 3: Weighted sum to get context vectors
        context_vecs = attn_weights @ values  # [seq_len, d_out]

        return context_vecs

torch.manual_seed(123)

inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your
    [0.55, 0.87, 0.66],  # Journey
    [0.57, 0.85, 0.64],  # Starts
    [0.22, 0.58, 0.33],  # With
    [0.77, 0.25, 0.10],  # One
    [0.05, 0.80, 0.55]   # Step
])

d_in = 3
d_out = 2

# Initialize self-attention module
sa_v1 = SelfAttention_v1(d_in, d_out)

# Forward pass
context_vectors = sa_v1(inputs)
print(context_vectors)
