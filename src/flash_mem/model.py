import numpy as np
import torch
from torch import nn
from torch import optim

num_features = 8
dataset = np.loadtxt("activations.csv", delimiter=",")
X = dataset[:, 0:num_features]
y = dataset[:, num_features]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(num_features, 12), nn.ReLU(), nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 1)
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    loss = 0
    for i in range(0, len(X), batch_size):
        Xbatch = X[i : i + batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i : i + batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f" epoch {epoch} loss {loss}")

with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")
