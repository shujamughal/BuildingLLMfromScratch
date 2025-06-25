import re
from torch.utils.data import Dataset, DataLoader

# 1. Load the text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. Tokenize using regex
tokens = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
tokens = [t.strip() for t in tokens if t.strip()]

# 3. Build vocabulary
vocab = sorted(set(tokens))
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

# 4. Convert to token IDs
input_ids = [token_to_id[t] for t in tokens]

# 5. Define Dataset class
class GPTDataset(Dataset):
    def __init__(self, input_ids, context_size):
        self.X = []
        self.Y = []
        for i in range(len(input_ids) - context_size):
            self.X.append(input_ids[i : i + context_size])
            self.Y.append(input_ids[i + 1 : i + 1 + context_size])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 6. Create dataset and dataloader
context_size = 4
dataset = GPTDataset(input_ids, context_size=context_size)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 7. Test a batch
for X_batch, Y_batch in loader:
    print("Input Batch:", X_batch)
    print("Target Batch:", Y_batch)
    break
# 8. Print first few examples
for i in range(3):
    print(f"Input (X[{i}]):   {dataset.X[i]}")
    print(f"Target (Y[{i}]):  {dataset.Y[i]}")
    print("---")
