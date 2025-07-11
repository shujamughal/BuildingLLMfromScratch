import torch

inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # "Your"    (x¹)
    [0.55, 0.87, 0.66],  # "Journey" (x²)
    [0.57, 0.85, 0.64],  # "Starts"  (x³)
    [0.22, 0.58, 0.33],  # "With"    (x⁴)
    [0.77, 0.25, 0.10],  # "One"     (x⁵)
    [0.05, 0.80, 0.55],  # "Step"    (x⁶)
])

query = inputs[1]  # [0.55, 0.87, 0.66]

attention_scores = torch.tensor([torch.dot(query, token) for token in inputs])
print(attention_scores)

query = inputs[1]  # Journey: [0.55, 0.87, 0.66]
target = inputs[0] # Your:    [0.43, 0.15, 0.89]

res = 0.
for idx in range(len(target)):
    res += target[idx] * query[idx]

print("Manual Dot Product:", res)
print("Using torch.dot:", torch.dot(target, query))

# option 1: Manual Normalization
attn_weights_simple = attention_scores/ attention_scores.sum()
print("Normalized Weights:", attn_weights_simple)
print("Sum:", attn_weights_simple.sum())

# option 2: Using Naive Softmax Implementation
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum()

attn_weights_naive = softmax_naive(attention_scores)
print("Naive Softmax Weights:", attn_weights_naive)
print("Sum:", attn_weights_naive.sum())

# option 3: Using PyTorch's Softmax
attn_weights_torch = torch.softmax(attention_scores, dim=0)
print("PyTorch Softmax Weights:", attn_weights_torch)
print("Sum:", attn_weights_torch.sum())

query = inputs[1]  # "journey"
context_vec_manual = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_manual += attn_weights_torch[i] * x_i

print("Context vector z(2):", context_vec_manual)

context_vec = torch.sum(attn_weights_torch[:, None] * inputs, dim=0)
print("Context vector:", context_vec)

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

print(attn_weights.sum(dim=-1))  # should print all 1.0


all_context_vectors = attn_weights @ inputs
print(all_context_vectors)

print("context_vec_2 (manual):", context_vec)
print("context_vec_2 (matrix):", all_context_vectors[1])
