import torch
from torch import nn

vocab_size = 50257       # Size of GPT-2 vocabulary
embedding_dim = 256      # Each token/position is a 256D vector
context_length = 4       # Number of tokens in each input sequence

token_embedding = nn.Embedding(vocab_size, embedding_dim)
print("Token Embedding:", token_embedding.weight.shape)
# Positional embedding for each position in the context length
positional_embedding = nn.Embedding(context_length, embedding_dim)
print("Positional Embedding:", positional_embedding.weight.shape)
batch_size = 8
inputs = torch.randint(0, vocab_size, (batch_size, context_length))
token_vectors = token_embedding(inputs)
print("Token Embeddings:", token_vectors.shape)
positions = torch.arange(context_length)
position_vectors = positional_embedding(positions)  # [4, 256]
final_embeddings = token_vectors + position_vectors
# Shape: [8, 4, 256]
print("Final Embeddings:", final_embeddings.shape)
# Final embeddings shape is [batch_size, context_length, embedding_dim]
print("Final Embeddings:", final_embeddings)
# Final embeddings are the sum of token and positional embeddings
# This allows the model to understand both the identity of tokens and their positions in the sequence   
# Final embeddings can be used as input to a transformer model
# or any other neural network that requires token embeddings with positional information    

