import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from models import SimpleCNN
from utils import get_filtered_dataset, plot_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and split dataset
dataset = get_filtered_dataset(train=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Model setup
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_acc, val_acc, train_loss, val_loss = [], [], [], []
best_acc = 0.0

for epoch in range(10):
    model.train()
    correct, total, running_loss = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc.append(correct / total)
    train_loss.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    correct, total, val_running_loss = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc.append(correct / total)
    val_loss.append(val_running_loss / len(val_loader))

    # Save best model
    if val_acc[-1] > best_acc:
        best_acc = val_acc[-1]
        os.makedirs("results", exist_ok=True)
        torch.save(model.state_dict(), "results/best_model.pth")

    print(f"Epoch {epoch+1}: Train Acc={train_acc[-1]:.2f}, Val Acc={val_acc[-1]:.2f}")

# Save metrics plot
plot_metrics(train_acc, val_acc, train_loss, val_loss)
