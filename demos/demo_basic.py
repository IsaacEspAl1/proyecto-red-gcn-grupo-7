import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd

# Create the connections between people (edges in the social graph)
connections = torch.tensor([
    [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 5, 6, 4, 7, 4, 7, 5]
], dtype=torch.long)

# Each person has 3 features: [age, fav_games, fav_art]
people_features = torch.tensor([
    [21, 1, 0],
    [22, 1, 0],
    [23, 1, 0],
    [24, 0, 1],
    [20, 0, 1],
    [21, 0, 1],
    [22, 1, 0],
    [23, 1, 0]
], dtype=torch.float)

# Here we have two class labels: 0 = Gamer, 1 = Artist
labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0], dtype=torch.long)

# Training and testing masks
train_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
test_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)

# Create the graph data
data = Data(people_features=people_features, connections=connections, labels=labels, train_mask=train_mask, test_mask=test_mask)

# Define the neural network model using GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 4)  # First layer: input size = 3, output size = 4
        self.conv2 = GCNConv(4, 2)  # Second layer: 4 → 2 (we have 2 classes)

    def forward(self, data):
        x, edge_index = data.people_features, data.connections
        x = F.relu(self.conv1(x, edge_index))  # Activation function ReLU
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Output: class probabilities

# Use GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


results = []

# Training loop
for epoch in range(101):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    
    # Calculate loss only with training data
    loss = F.nll_loss(out[data.train_mask], data.labels[data.train_mask])
    loss.backward()
    optimizer.step()

    # Evaluation of the model
    model.eval()
    predictions = out.argmax(dim=1)
    correct = int((predictions[data.test_mask] == data.labels[data.test_mask]).sum())
    acc = correct / int(data.test_mask.sum())

    # Save current epoch results
    results.append({
        'epoch': epoch,
        'loss': loss.item(),
        'accuracy': acc
    })

    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

# Save all training results to CSV (used later to create plot)
df = pd.DataFrame(results)
df.to_csv("resultados_entrenamiento_red_social.csv", index=False)

