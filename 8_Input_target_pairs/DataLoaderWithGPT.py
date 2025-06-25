from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader

# 1. Load GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# 2. Load and encode book text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

input_ids = tokenizer.encode(text, add_special_tokens=False)

# 3. Prepare Dataset
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

# 4. Wrap in DataLoader
context_size = 8
batch_size = 16

dataset = GPTDataset(input_ids, context_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Test a batch
for X_batch, Y_batch in loader:
    print("Encoded Input IDs:", X_batch)
    print("Encoded Target IDs:", Y_batch)
    break

# 5. Preview a batch
for X_batch, Y_batch in loader:
    print("Input:", tokenizer.batch_decode(X_batch))
    print("Target:", tokenizer.batch_decode(Y_batch))
    break




