import torch
import torch.nn as nn

# 🧠 Custom GELU activation function (used in GPT)
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * x.pow(3))
        ))

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        # 🔧 Create layers: Linear layer followed by GELU
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                GELU()
            )
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)

            # ➕ Add shortcut if:
            #   1. The user enabled it
            #   2. The input and output shapes match
            if self.use_shortcut and out.shape == x.shape:
                x = x + out  # 🛣️ Add shortcut path
            else:
                x = out  # 🧱 Standard forward pass
        return x

# Example usage
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])  # shape: [1, 3]

# Set seed for reproducibility
torch.manual_seed(123)

# 🧱 Deep model without shortcuts
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

# 🛣️ Deep model with residual (shortcut) connections
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)

def print_gradients(model, x):
    output = model(x)  # Forward pass
    target = torch.tensor([[0.]])  # Dummy target

    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)  # Compute loss
    loss.backward()  # Backpropagation

    # Print average absolute gradient for each Linear layer
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} ➤ Gradient mean: {param.grad.abs().mean().item():.6f}")

print("🚫 Without Shortcut Connections:")
print_gradients(model_without_shortcut, sample_input)

print("✅ With Shortcut Connections:")
print_gradients(model_with_shortcut, sample_input)
