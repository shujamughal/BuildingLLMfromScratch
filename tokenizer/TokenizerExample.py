# step 1
text = "Hello, world! This is a test."

# Step 2: Tokenization using Regex
import re

tokens = re.split(r"(\s+|,|\.|:|;|\?|\!|\"|--|\(|\))", text)
tokens = [t for t in tokens if t.strip()]
print("Tokens:", tokens)

# Step 3: Build Vocabulary (Token → ID)
vocab = sorted(set(tokens))
token_to_id = {token: idx for idx, token in enumerate(vocab)}

# Add special tokens
token_to_id["<unk>"] = len(token_to_id)
token_to_id["<eot>"] = len(token_to_id)
# Step 4: Encode Tokens (Map Tokens to Token IDs)
token_ids = [token_to_id[token] for token in tokens]
print("Token IDs:", token_ids)
# Step 5: Decode Token IDs (Map Token IDs back to Tokens)
id_to_token = {idx: token for token, idx in token_to_id.items()}
decoded_tokens = [id_to_token[id_] for id_ in token_ids]
decoded_text = " ".join(decoded_tokens)
print("Decoded text:", decoded_text)

# Step 6: Use SimpleTokenizer for encoding and decoding
from SimpleTokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer(token_to_id)

text = "Hello, do you like tea? <eot> In the sunlit terraces of the palace."
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print("Token IDs:", ids)
print("Decoded Text:", decoded)
