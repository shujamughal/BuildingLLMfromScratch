import torch.nn as nn
import torch

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])
    
    def forward(self, x):
        # Concatenate all head outputs along the last dimension
        return torch.cat([head(x) for head in self.heads], dim=-1)


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

batch = torch.stack((inputs, inputs), dim=0)  # shape: [2, 6, 3]

d_in = 3
d_out = 2
context_length = batch.shape[1]
dropout = 0.0
num_heads = 2

mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
context_vecs = mha(batch)

print("context_vecs:", context_vecs)
print("context_vecs.shape:", context_vecs.shape)
