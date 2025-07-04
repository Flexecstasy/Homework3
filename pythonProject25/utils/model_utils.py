import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import logging

class FullyConnectedNet(nn.Module):
    def __init__(self, layer_sizes, dropout=None, batchnorm=False):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if batchnorm:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        # Последний линейный слой без ReLU
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model.to(device)
    for epoch in range(1, epochs+1):
        # Train
        model.train()
        train_losses, train_preds, train_labels = [], [], []
        for x, y in dataloaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = out.argmax(dim=1).cpu().tolist()
            train_preds.extend(preds)
            train_labels.extend(y.cpu().tolist())
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for x, y in dataloaders['val']:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_losses.append(loss.item())
                preds = out.argmax(dim=1).cpu().tolist()
                val_preds.extend(preds)
                val_labels.extend(y.cpu().tolist())
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = accuracy_score(val_labels, val_preds)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    return history