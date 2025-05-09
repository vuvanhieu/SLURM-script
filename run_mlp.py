import torch
import torch.nn as nn

print("✅ [MLP] PyTorch version:", torch.__version__)
print("✅ [MLP] CUDA available:", torch.cuda.is_available())

class SimpleMLP(nn.Module):
    def __init__(self, input_size=100, hidden_size=64, output_size=10):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleMLP()
dummy_input = torch.randn(1, 100)
output = model(dummy_input)

print("✅ [MLP] Model ran successfully. Output shape:", output.shape)
