import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # γ
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
torch.manual_seed(123)
batch = torch.randn(2, 5)

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch)

print("LayerNorm Output:\n", out_ln)
print("Mean:\n", out_ln.mean(dim=-1, keepdim=True))
print("Variance:\n", out_ln.var(dim=-1, unbiased=False, keepdim=True))
