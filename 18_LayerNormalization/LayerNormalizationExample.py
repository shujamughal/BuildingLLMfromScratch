import torch
import torch.nn as nn

torch.manual_seed(123)
batch = torch.randn(2, 5)  # 2 inputs, each with 5 features
print(batch)

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
norm_out = (out - mean) / torch.sqrt(var + 1e-5)
print("Normalized:\n", norm_out)
print("New Mean:\n", norm_out.mean(dim=-1, keepdim=True))
print("New Variance:\n", norm_out.var(dim=-1, keepdim=True))

