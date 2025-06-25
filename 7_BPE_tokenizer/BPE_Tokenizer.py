import tiktoken

# Load GPT-2’s tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Sample text
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."

# Allow the special token
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print("Token IDs:", token_ids)

# Decode
decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)

text = "akwirwier"
token_ids = tokenizer.encode(text)
print(token_ids)
print(tokenizer.decode(token_ids))
