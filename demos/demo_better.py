}import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd

# People features: [age, likes_games, fav_art, fav_sports]
# This matrix represents 12 people and what they like
people_features = torch.tensor([
    [21, 1, 0, 0],
    [22, 1, 0, 0],
    [23, 1, 0, 0],
    [24, 0, 1, 0],
    [20, 0, 1, 0],
    [21, 0, 1, 0],
    [25, 0, 0, 1],
    [26, 0, 0, 1],
    [27, 0, 0, 1],
    [22, 1, 0, 1],
    [23, 0, 1, 1],
    [24, 1, 1, 0],
], dtype=torch.float)

# Class for each person: 0 = Gamer, 1 = Artist, 2 = Athlete
labels = torch.tensor([
    0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    2, 1, 0
], dtype=torch.long)

# Connections between people (edges)
# Each pair represents a friendship
connections = torch.tensor([
    [0, 1, 1, 2, 2, 9, 9, 6, 6, 7, 7, 8, 3, 4, 4, 5, 5, 10, 10, 11, 11, 0, 11],
    [1, 0, 2, 1, 9, 2, 6, 9, 7, 6, 8, 7, 4, 3, 5, 4, 10, 5, 11, 10, 0, 11, 10]
], dtype=torch.long)

# Training and test masks
# We tell the model which people to use for training and which to use for testing
train_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], dtype=torch.bool)
test_mask = ~train_mask


data = Data(people_features=people_features, connections=connections, labels=labels, train_mask=train_mask, test_mask=test_mask)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # First layer: input 4 features, output 8
        self.conv1 = GCNConv(4, 8)
        # Second layer: from 8 to 3 classes
        self.conv2 = GCNConv(8, 3)

    def forward(self, data):
        x, edge_index = data.people_features, data.connections
        x = F.relu(self.conv1(x, edge_index))         # Apply first layer + activation
        x = F.dropout(x, training=self.training, p=0.3) # Dropout to avoid overfitting
        x = self.conv2(x, edge_index)                 # Apply second layer
        return F.log_softmax(x, dim=1)                # Final output: probabilities

# Use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)

# Optimizer to update the weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


results = []

# Training loop (100 times)
for epoch in range(101):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.labels[data.train_mask])
    loss.backward()
    optimizer.step()

    # Evaluation of the model
    model.eval()
    pred = out.argmax(dim=1)
    correct = int((pred[data.test_mask] == data.labels[data.test_mask]).sum())
    total = int(data.test_mask.sum())
    acc = correct / total

    results.append({
        'epoch': epoch,
        'loss': loss.item(),
        'accuracy': acc
    })

    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

# Save results to a CSV file (do not change file name)
df = pd.DataFrame(results)
df.to_csv("resultados_entrenamiento_red_social_mejorada.csv", index=False)


