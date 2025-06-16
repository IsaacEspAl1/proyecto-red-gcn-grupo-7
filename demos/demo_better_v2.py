import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
import matplotlib.pyplot as plt
import pandas as pd

# Node info: [age, fav_games, fav_music, fav_sports]
people_features = torch.tensor([
    [21, 1, 0, 0], [22, 1, 0, 0], [23, 1, 0, 0],
    [24, 0, 1, 0], [20, 0, 1, 0], [21, 0, 1, 0],
    [25, 0, 0, 1], [26, 0, 0, 1], [27, 0, 0, 1],
    [22, 1, 0, 1], [23, 0, 1, 1], [24, 1, 1, 0],
], dtype=torch.float)

# Labels also 3: 0 = gamer, 1 = musician, 2 = athlete
labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 0], dtype=torch.long)

# Node connections (edges)
connections = torch.tensor([
    [0, 1, 1, 2, 2, 9, 9, 6, 6, 7, 7, 8, 3, 4, 4, 5, 5, 10, 10, 11, 11, 0, 11],
    [1, 0, 2, 1, 9, 2, 6, 9, 7, 6, 8, 7, 4, 3, 5, 4, 10, 5, 11, 10, 0, 11, 10]
], dtype=torch.long)

# Select train and test nodes
train_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], dtype=torch.bool)
test_mask = ~train_mask

# Create the graph data
data = Data(people_features=people_features, connections=connections, labels=labels, train_mask=train_mask, test_mask=test_mask)

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(4, 16)
        self.bn1 = BatchNorm(16)
        self.conv2 = GCNConv(16, 8)
        self.bn2 = BatchNorm(8)
        self.conv3 = GCNConv(8, 3)

    def forward(self, data):
        x, edge_index = data.people_features, data.connections
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


results = []
final_predictions = None

# Training loop
for epoch in range(201):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.labels[data.train_mask])
    loss.backward()
    optimizer.step()

    
    model.eval()
    pred = out.argmax(dim=1)
    correct = int((pred[data.test_mask] == data.labels[data.test_mask]).sum())
    total = int(data.test_mask.sum())
    accuracy = correct / total

    results.append({'epoch': epoch, 'loss': loss.item(), 'accuracy': accuracy})

    if epoch == 200:
        final_predictions = pred.cpu().numpy()


df = pd.DataFrame(results)
df.to_csv("resultados_entrenamiento_red_social_mejorada_v2.csv", index=False)

