import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

print("✅ [GNN] PyTorch version:", torch.__version__)
print("✅ [GNN] CUDA available:", torch.cuda.is_available())

# Dummy graph with 3 nodes and 2 edges
x = torch.tensor([[1], [2], [3]], dtype=torch.float)  # node features
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index)

class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(1, 4)

    def forward(self, data):
        return self.conv1(data.x, data.edge_index)

model = SimpleGNN()
output = model(data)

print("✅ [GNN] Model ran successfully. Output shape:", output.shape)
