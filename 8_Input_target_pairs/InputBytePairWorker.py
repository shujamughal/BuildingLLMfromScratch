# A simulated tokenized sequence (e.g., from a sentence or paragraph)
token_ids = [290, 4920, 2241, 287, 257, 300, 345, 421, 112, 89]

context_size = 4

for i in range(1, context_size+1):
    context = token_ids[:i]
    desired = token_ids[i]

    print(context, "---->", desired)

print("--------------------------------------")


X = []  # input
Y = []  # target (next-token predictions)

# Loop to slide context window to show how the model learns
# to predict the next token based on the context
for i in range(len(token_ids) - context_size):
    x = token_ids[i:i + context_size]
    y = token_ids[i + 1:i + 1 + context_size]
    X.append(x)
    Y.append(y)

# Print out first few examples
for i in range(3):
    print(f"Input (X[{i}]):   {X[i]}")
    print(f"Target (Y[{i}]):  {Y[i]}")
    print("---")
