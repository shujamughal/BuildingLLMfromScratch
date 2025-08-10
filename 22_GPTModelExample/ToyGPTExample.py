import torch, math
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# Toy config (easy to print)
B, T, D, V, n_heads = 1, 4, 8, 20, 2
head_dim = D // n_heads

# Simple GELU and LayerNorm (toy)
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

class ToyLayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

# Toy MHA (explicit, for education)
class ToyMHA(nn.Module):
    def __init__(self, D, n_heads, dropout=0.0):
        super().__init__()
        self.D = D
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        self.Wout = nn.Linear(D, D)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,D = x.shape
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, T, head_dim)
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        # causal mask (allow attending to current and past positions)
        mask = torch.tril(torch.ones(T, T)).to(scores.device)  # (T, T)
        scores = scores.masked_fill(mask[None, None, :, :] == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)  # (B, n_heads, T, T)
        attn = self.dropout(attn)
        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1,2).contiguous().view(B, T, D)  # (B, T, D)
        return self.Wout(out)  # (B, T, D)

class ToyFFN(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 4*D),
            GELU(),
            nn.Linear(4*D, D),
        )
    def forward(self, x):
        return self.net(x)

# Toy Transformer Block
class ToyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = ToyLayerNorm(cfg["emb_dim"])
        self.att  = ToyMHA(cfg["emb_dim"], cfg["n_heads"], dropout=cfg["drop_rate"])
        self.norm2 = ToyLayerNorm(cfg["emb_dim"])
        self.ff = ToyFFN(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut   # residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut   # residual
        return x

# Toy GPT-like pipeline
cfg = {"vocab_size": V, "context_length": 1024, "emb_dim": D, "n_heads": n_heads, "n_layers": 1, "drop_rate": 0.0, "qkv_bias": False}

# 1) Input tokens (toy)
in_idx = torch.tensor([[2, 5, 3, 7]])   # shape (B,T)
print("Step 1 - token ids:", in_idx.shape)

# 2) Token & Positional embeddings
tok_emb = nn.Embedding(V, D)
pos_emb = nn.Embedding(1024, D)
tok = tok_emb(in_idx)                        # (B,T,D)
pos = pos_emb(torch.arange(T))               # (T,D)
x = tok + pos                                # broadcasted to (B,T,D)
print("Step 2-3 - token+pos embeddings:", x.shape)

# 4) dropout (no-op here)
drop = nn.Dropout(0.0)
x = drop(x)
print("Step 4 - after dropout:", x.shape)

# 5) Transformer blocks
block = ToyTransformerBlock(cfg)
x = block(x)  # shape preserved
print("Step 5 - after Transformer block:", x.shape)

# 6) Final LayerNorm
final_norm = ToyLayerNorm(D)
x = final_norm(x)
print("Step 6 - after final LayerNorm:", x.shape)

# 7) Output head -> logits
out_head = nn.Linear(D, V, bias=False)
logits = out_head(x)  # (B, T, V)
print("Step 7 - logits shape:", logits.shape)
