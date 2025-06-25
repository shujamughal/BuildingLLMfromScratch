import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Step 1: Dataset class
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Step 2: DataLoader creator
def create_dataloader_v1(txt, batch_size=2, max_length=4, 
                         stride=1, shuffle=False, drop_last=False,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader

# Step 3: Try with small fake text
text = "Hello, this is a simple demo of sliding window for LLM."

# Create dataloader
loader = create_dataloader_v1(text, batch_size=1, max_length=4, stride=1)

# Step 4: Preview first few batches
tokenizer = tiktoken.get_encoding("gpt2")
print("Decoded Input/Target Batches:\n")

for i, (input_ids, target_ids) in enumerate(loader):
    print(f"Batch {i+1}")
    print("Input IDs :", input_ids)
    print("Target IDs:", target_ids)
    print("Input Text :", tokenizer.decode(input_ids[0].tolist()))
    print("Target Text:", tokenizer.decode(target_ids[0].tolist()))
    print("-" * 40)
    if i == 3: break  # Limit to first 4 batches
