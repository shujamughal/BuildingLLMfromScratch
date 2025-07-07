import torch
import torch.nn as nn

# Step 1: Create an embedding layer
embedding_layer = nn.Embedding(num_embeddings=5, embedding_dim=3)

print(embedding_layer.weight)


# Step 2: Select input token IDs (e.g., "banana" → 1, "car" → 3)
#input_ids = torch.tensor([1, 3])  # Shape: (2,)
input_ids = torch.tensor([[1, 3], [4, 2]])

# Step 3: Lookup vectors
output_vectors = embedding_layer(input_ids)


print("Input IDs:", input_ids)
print("Embedding Vectors:\n", output_vectors)
# Step 4: View the shape of the output vectors
print("Output Vectors Shape:", output_vectors.shape)  # Should be (2,