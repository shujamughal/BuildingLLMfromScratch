import torch
import torch.nn as nn
import math

# 1️⃣ GELU Activation (used in Transformer FFN)
class GELU(nn.Module):
    def forward(self, x):
        # Efficient approximation used in GPT-2
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) *
            (x + 0.044715 * x.pow(3))
        ))

# 2️⃣ Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))  # γ
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # per token (last dim)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift  # learnable scale and shift

# 3️⃣ FeedForward Network inside Transformer block
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expand 768 → 3072
            GELU(),                                         # Apply GELU activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # Compress back to 768
        )

    def forward(self, x):
        return self.layers(x)

# 4️⃣ Multi-Head Attention with masking (causal)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Shared Linear projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Upper-triangular causal mask
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Project and reshape: [b, tokens, d_out] → [b, heads, tokens, head_dim]
        keys    = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [b, heads, tokens, tokens]
        attn_scores = queries @ keys.transpose(2, 3)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask, -torch.inf)

        # Softmax → Dropout → Attention-weighted values
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = attn_weights @ values  # [b, heads, tokens, head_dim]

        # Combine heads: [b, tokens, d_out]
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)

# 5️⃣ Transformer Block (Attention + FFN + LayerNorm + Skip)
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention block with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # FeedForward block with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# 6️⃣ Sample GPT-2 configuration dictionary
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 7️⃣ Run the transformer block on dummy input
if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)  # 2 samples, 4 tokens each, 768-dimensional embeddings
    block = TransformerBlock(GPT_CONFIG_124M)
    out = block(x)

    print("Input shape: ", x.shape)
    print("Output shape:", out.shape)
