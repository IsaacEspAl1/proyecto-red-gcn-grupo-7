import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv("dataset_red_social_ampliada.csv")


features = df[["edad", "fav_games", "fav_art", "fav_sports", "fav_academic"]].values
labels = df["group"].values
nodes = df["node"].values


G = nx.Graph()
G.add_nodes_from(nodes)
for i, node_i in enumerate(nodes):
    for j, node_j in enumerate(nodes):
        if i != j:
            f_i = features[i]
            f_j = features[j]
            # Connect nodes if they're similar enough
            if np.linalg.norm(f_i - f_j) < 8:
                G.add_edge(node_i, node_j)

# Create edge list with index
node_to_index = {node: i for i, node in enumerate(nodes)}
edges = []
for u, v in G.edges():
    edges.append([node_to_index[u], node_to_index[v]])
    edges.append([node_to_index[v], node_to_index[u]])

# Convert to torch tensors
connections = torch.tensor(edges, dtype=torch.long).t().contiguous()
node_features = torch.tensor(features, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.long)

# Split data into training and test
train_idx, test_idx = train_test_split(range(len(nodes)), test_size=0.3, stratify=labels, random_state=42)
train_mask = torch.zeros(len(nodes), dtype=torch.bool)
test_mask = torch.zeros(len(nodes), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

# Create the PyG data object
data = Data(people_features=node_features, connections=connections, labels=labels, train_mask=train_mask, test_mask=test_mask)

# Define the Graph Convolutional Network
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.people_features, data.connections
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(input_dim=5, hidden_dim=16, output_dim=4).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
results = []
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.labels[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    _, pred = out[data.test_mask].max(dim=1)
    correct = pred.eq(data.labels[data.test_mask]).sum().item()
    accuracy = correct / data.test_mask.sum().item()
    results.append((epoch, loss.item(), accuracy))


df_resultados = pd.DataFrame(results, columns=["epoch", "loss", "accuracy"])
df_resultados.to_csv("resultados_entrenamiento_red_social_avanzada.csv", index=False)

print("Training completed. Results saved in 'resultados_entrenamiento_red_social_avanzada.csv'")
