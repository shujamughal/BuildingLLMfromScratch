# Step 1 Read the Full Text

with open("the-verdict.txt", "r") as f:
    raw_text = f.read()
print("Total characters:", len(raw_text))
print(raw_text[:100])  # Preview

# Step 2 Tokenize Full Dataset
import re
preprocessed = re.split(r"(\s+|,|\.|:|;|\?|\!|\"|--|\(|\))", raw_text)
preprocessed = [t for t in preprocessed if t.strip()]
print("Sample Tokens:", preprocessed[:30])

# step 3 Build Vocabulary from Dataset
vocab = sorted(set(preprocessed))
token_to_id = {token: idx for idx, token in enumerate(vocab)}

# Add special tokens
token_to_id["<unk>"] = len(token_to_id)
token_to_id["<eot>"] = len(token_to_id)

# Step 4 Initialize Tokenizer and Test
from SimpleTokenizer import SimpleTokenizer
tokenizer = SimpleTokenizer(token_to_id)

sample = "This is not in the original line. <eot> Here is a line from the book."
ids = tokenizer.encode(sample)
print("Token IDs:", ids)

decoded = tokenizer.decode(ids)
print("Decoded Text:", decoded)
