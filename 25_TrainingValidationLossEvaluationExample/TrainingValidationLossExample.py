import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken

from FullGPTExampleWithTextGeneration import GPTModel

# ---------------------------
# 1. Load dataset (text)
# ---------------------------
try:
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
except FileNotFoundError:
    # fallback demo text
    text_data = "Every effort moves you forward. I really like chocolate. Language models learn by predicting the next token."

# ---------------------------
# 2. Tokenizer (GPT-2 encoding via tiktoken)
# ---------------------------
tokenizer = tiktoken.get_encoding("gpt2")

# Convert whole text into **token IDs**
encoded = tokenizer.encode(text_data)
print("Total characters:", len(text_data))
print("Total tokens:", len(encoded))

# ---------------------------
# 3. Split into train/val sets (now numeric IDs, not raw text)
# ---------------------------
train_ratio = 0.9
split_idx = int(train_ratio * len(encoded))
train_data = encoded[:split_idx]
val_data = encoded[split_idx:]

# ---------------------------
# 4. Dataset: now works directly on token IDs
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, data_ids, max_length, stride):
        self.data_ids = data_ids
        self.max_length = max_length
        self.stride = stride
        self.samples = []

        # Create overlapping windows of token IDs
        for i in range(0, len(data_ids) - max_length, stride):
            x = data_ids[i:i+max_length]
            y = data_ids[i+1:i+1+max_length]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ---------------------------
# 5. Re-implement create_dataloader_v1
# ---------------------------
def create_dataloader_v1(data_ids, batch_size, max_length, stride, drop_last, shuffle):
    dataset = TextDataset(data_ids, max_length=max_length, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return loader

# ---------------------------
# 6. Create loaders
# ---------------------------
torch.manual_seed(123)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # shorter for speed
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False
)

# ---------------------------
# 7. Check a batch
# ---------------------------
print("Train loader:")
for x, y in train_loader:
    print("x:", x.shape, "y:", y.shape)
    print("Example x[0]:", x[0][:10])  # first 10 token IDs
    print("Example y[0]:", y[0][:10])
    break

print("\nValidation loader:")
for x, y in val_loader:
    print("x:", x.shape, "y:", y.shape)
    break

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),  # [batch*tokens, vocab]
        target_batch.flatten() # [batch*tokens]
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.eval()

model.to(device)

with torch.no_grad():  # no gradients during eval
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss   = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)
