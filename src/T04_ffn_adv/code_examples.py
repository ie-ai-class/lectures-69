"""
code_examples.py  —  T04: Advanced FFN Training Techniques
AI Class, CMU-NR

Companion code for Sections 4.1 – 4.5.
Each section is self-contained and can be run independently.
Requires: torch >= 2.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# Shared: tiny XOR dataset (from T03)
# ─────────────────────────────────────────────────────────────────────────────

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])


# =============================================================================
# Section 4.1 — L2 and L1 Regularisation
# =============================================================================

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# --- L2 regularisation: built into the optimiser as weight_decay ---

def train_l2(weight_decay=0.01, epochs=1000):
    model = SimpleNet()
    # weight_decay adds λ·Σw² to the loss automatically
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    print(f"[L2 weight_decay={weight_decay}] Final loss: {loss.item():.4f}")


# --- L1 regularisation: added manually to the loss ---

def train_l1(lambda_l1=0.01, epochs=1000):
    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)

        # Add L1 penalty: λ · Σ|w|
        l1_penalty = lambda_l1 * sum(p.abs().sum() for p in model.parameters())
        (loss + l1_penalty).backward()
        optimizer.step()

    print(f"[L1 lambda={lambda_l1}] Final loss: {loss.item():.4f}")


# =============================================================================
# Section 4.2 — Dropout
# =============================================================================

class NetWithDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.fc1     = nn.Linear(2, 2)
        self.drop    = nn.Dropout(p=p)   # applied after hidden activation
        self.fc2     = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop(x)               # active only in training mode
        x = torch.sigmoid(self.fc2(x))
        return x


def train_dropout(p=0.5, epochs=2000):
    model = NetWithDropout(p=p)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()                  # dropout ON
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    model.eval()                       # dropout OFF
    with torch.no_grad():
        preds = model(X)
    print(f"[Dropout p={p}] Predictions: {preds.squeeze().tolist()}")


# =============================================================================
# Section 4.3 — Early Stopping
# =============================================================================

def train_early_stopping(patience=4, max_epochs=200):
    """
    Saves the best checkpoint and restores it when validation loss
    has not improved for `patience` consecutive epochs.
    """
    # Split: 3 training samples, 1 validation (tiny demo)
    X_train, y_train = X[:3], y[:3]
    X_val,   y_val   = X[3:], y[3:]

    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_weights  = None
    wait          = 0            # epochs without improvement

    for epoch in range(1, max_epochs + 1):
        # --- training step ---
        model.train()
        optimizer.zero_grad()
        train_loss = criterion(model(X_train), y_train)
        train_loss.backward()
        optimizer.step()

        # --- validation step ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = " ← best"
        else:
            wait += 1
            marker = f" (patience {wait}/{patience})"

        if epoch <= 10 or epoch % 20 == 0:
            print(f"Epoch {epoch:3d}  train={train_loss.item():.4f}  "
                  f"val={val_loss:.4f}{marker}")

        if wait >= patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Restoring weights from best epoch.")
            model.load_state_dict(best_weights)
            break

    return model


# =============================================================================
# Section 4.4 — Batch Normalisation
# =============================================================================

class NetWithBN(nn.Module):
    """
    BN is placed between the linear layer and the activation.
    bias=False in fc1 because BN subtracts the batch mean anyway.
    """
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(2, 4, bias=False)   # bias redundant with BN
        self.bn1  = nn.BatchNorm1d(4)
        self.fc2  = nn.Linear(4, 1)

    def forward(self, x):
        x = self.bn1(self.fc1(x))    # normalise pre-activations
        x = torch.sigmoid(x)
        x = torch.sigmoid(self.fc2(x))
        return x


def train_batch_norm(epochs=2000):
    # Need a proper mini-batch loader for BN to work correctly
    dataset    = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model     = NetWithBN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()                  # BN uses mini-batch statistics
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()                       # BN uses running statistics
    with torch.no_grad():
        preds = model(X)
    print(f"[BatchNorm] Predictions: {preds.squeeze().tolist()}")


# =============================================================================
# Section 4.5 — Weight Initialisation
# =============================================================================

class NetExplicitInit(nn.Module):
    def __init__(self, activation="sigmoid"):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.activation = activation
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            if self.activation in ("sigmoid", "tanh"):
                nn.init.xavier_uniform_(layer.weight)   # Xavier / Glorot
            else:
                nn.init.kaiming_normal_(layer.weight)   # He (for ReLU)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        if self.activation == "relu":
            x = torch.relu(self.fc1(x))
        else:
            x = torch.sigmoid(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# =============================================================================
# Section 4.5 — Learning Rate Schedules
# =============================================================================

def train_with_lr_schedule(schedule="step", epochs=100):
    model     = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()

    if schedule == "step":
        # Halve the LR every 10 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif schedule == "exponential":
        # Multiply LR by 0.95 every epoch
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()           # update LR after each epoch

        current_lr = optimizer.param_groups[0]['lr']
        if epoch in (1, 10, 20, 30, 50, 100):
            print(f"Epoch {epoch:3d}  loss={loss.item():.4f}  lr={current_lr:.6f}")


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("4.1  L2 Regularisation")
    print("=" * 60)
    train_l2(weight_decay=0.0,  epochs=1000)
    train_l2(weight_decay=0.01, epochs=1000)

    print("\n" + "=" * 60)
    print("4.1  L1 Regularisation")
    print("=" * 60)
    train_l1(lambda_l1=0.0,  epochs=1000)
    train_l1(lambda_l1=0.01, epochs=1000)

    print("\n" + "=" * 60)
    print("4.2  Dropout")
    print("=" * 60)
    train_dropout(p=0.0, epochs=2000)
    train_dropout(p=0.5, epochs=2000)

    print("\n" + "=" * 60)
    print("4.3  Early Stopping")
    print("=" * 60)
    train_early_stopping(patience=4, max_epochs=200)

    print("\n" + "=" * 60)
    print("4.4  Batch Normalisation")
    print("=" * 60)
    train_batch_norm(epochs=2000)

    print("\n" + "=" * 60)
    print("4.5  LR Schedule — Step Decay")
    print("=" * 60)
    train_with_lr_schedule(schedule="step", epochs=100)

    print("\n" + "=" * 60)
    print("4.5  LR Schedule — Exponential Decay")
    print("=" * 60)
    train_with_lr_schedule(schedule="exponential", epochs=100)
