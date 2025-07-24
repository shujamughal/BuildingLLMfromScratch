import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,       # Number of tokens in vocabulary
    "context_length": 1024,    # Maximum sequence length
    "emb_dim": 768,            # Embedding dimension per token
    "n_heads": 12,             # Number of attention heads
    "n_layers": 12,            # Transformer blocks stacked
    "drop_rate": 0.1,          # Dropout rate
    "qkv_bias": False          # Whether Q/K/V projections include a bias
}
# Placeholder Transformer Block - we'll replace this later
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x  # Just passes input forward for now

# Placeholder LayerNorm
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
import torch
import tiktoken
import torch.nn as nn

# Initialize GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
# Two example sentences
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# Tokenize both
batch = [
    torch.tensor(tokenizer.encode(txt1)),
    torch.tensor(tokenizer.encode(txt2))
]
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
