import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# model
class HeartModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# load data
df = pd.read_csv("Heart_disease_cleveland_new.csv")

X = df.drop("target", axis=1).values
y = df["target"].values


# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)

X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# dataloaders
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# setup model
input_dim = X_train.shape[1]
model = HeartModel(input_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train loop (MNISt)
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    seen = 0   # 👈 track progress correctly

    for batch, (X, y) in enumerate(dataloader):
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        seen += len(X)  # 👈 accumulate samples seen

        if batch % 10 == 0:
            print(f"loss: {loss.item():.6f}  [{seen:>5d}/{len(dataloader.dataset):>5d}]")

    return total_loss / len(dataloader)

# test loop
def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            preds = model(X)
            test_loss += loss_fn(preds, y).item()

            probs = torch.sigmoid(preds)
            predicted_class = (probs > 0.5).float()
            
            correct += (predicted_class == y).sum().item()

    accuracy = correct / len(dataloader.dataset)
    avg_loss = test_loss / len(dataloader)

    print(f"Test Error:\n Accuracy: {accuracy*100:.1f}%, Avg loss: {avg_loss:.6f}\n")

    return accuracy, avg_loss

# Training
epochs = 20
train_losses = []
test_losses = []
test_accuracies = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    
    train_loss = train_loop(train_loader, model, criterion, optimizer)
    acc, test_loss = test_loop(test_loader, model, criterion)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(acc)

print("Done!")

# sample prediction (MNIST)
sample_num = 5

model.eval()
with torch.no_grad():
    logits = model(X_test[sample_num].unsqueeze(0))
    prob = torch.sigmoid(logits)

print("\n--- Sample Prediction ---")
print("Raw model output (logit):", logits.item())
print("Predicted probability:", prob.item())
print("Predicted class:", int(prob.item() > 0.5))
print("Actual class:", int(y_test[sample_num].item()))


# plot
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()