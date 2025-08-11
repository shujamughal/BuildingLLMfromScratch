# Full GPT-like implementation (toy and real config demo)
import torch
import torch.nn as nn
import math
torch.manual_seed(123)

# -------------------------
# Building blocks
# -------------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x.pow(3))))

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            GELU(),
            nn.Linear(4*emb_dim, emb_dim),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # causal mask (1s above diagonal)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, _ = x.shape
        keys    = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)
        values  = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)   # (B, n_heads, T, T)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # (T, T)
        attn_scores.masked_fill_(mask_bool, -float('inf'))

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ values                       # (B, n_heads, T, head_dim)
        context = context.transpose(1,2).contiguous().view(b, num_tokens, self.d_out)
        out = self.out_proj(context)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"],
            qkv_bias=cfg.get("qkv_bias", False)
        )
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# -------------------------
# GPTModel (assembly)
# -------------------------
class GPTModel(nn.Module):
    def __init__(self, cfg, tie_weights=False):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if tie_weights:
            # tie (share) token embedding weights and output head weights (reduces param count)
            self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # (B, T, D)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (T, D)
        x = tok_embeds + pos_embeds  # broadcast -> (B, T, D)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (B, T, V)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text one token at a time using the GPT model.
    
    Parameters:
    -----------
    model : GPTModel
        The trained or untrained GPT model.
    idx : torch.Tensor
        Starting token IDs, shape [batch_size, current_length].
    max_new_tokens : int
        Number of new tokens to generate.
    context_size : int
        Model's maximum context length (memory window).
    """
    model.eval()  # 1. Disable training-specific behavior (like dropout)
    
    for _ in range(max_new_tokens):
        # 2. Keep only the last `context_size` tokens to avoid exceeding model limits
        idx_cond = idx[:, -context_size:]  # Shape: [batch, context_size]

        # 3. Forward pass — get model predictions
        with torch.no_grad():
            logits = model(idx_cond)  # Shape: [batch, seq_len, vocab_size]

        # 4. Take only the predictions for the last time step
        logits_last = logits[:, -1, :]  # Shape: [batch, vocab_size]

        # 5. Convert raw scores (logits) to probabilities
        probas = torch.softmax(logits_last, dim=-1)  # Shape: [batch, vocab_size]

        # 6. Pick the token with the highest probability (Greedy decoding)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # Shape: [batch, 1]

        # 7. Append this predicted token to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # Shape: [batch, current_length+1]
    
    return idx



# -------------------------
# GPT-2 small config (use carefully on CPU)
# -------------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
# If you want to build the full GPT-2 small model (takes memory), uncomment:
model_big = GPTModel(GPT_CONFIG_124M, tie_weights=False)
print("GPT-2 small params:", sum(p.numel() for p in model_big.parameters()))
gpt_batch = torch.tensor([[2,5,3,7]])
gpt_logits = model_big(gpt_batch)
print("GPT logits shape:", gpt_logits.shape)  # expect (1,4,20)

import tiktoken
# Load GPT-2 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
# Starting text
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)             # e.g., [15496, 11, 314, 716]
encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # Shape: [1, 4]

# Generate 6 new tokens
out = generate_text_simple(
    model=model_big,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

# Convert token IDs back to text
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)



