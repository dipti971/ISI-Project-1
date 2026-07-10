"""Training loop for GAT edge classifier.

Full training pipeline with:
- Class-weighted CrossEntropyLoss
- Adam optimizer with weight decay
- Early stopping (patience-based)
- Epoch-level logging (loss, accuracy)
- Test evaluation + prediction saving

Usage:
    python src/models/train_gat.py
"""

import os
import sys
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.gat_model import build_gat_model

# ==========================
# Configuration
# ==========================

GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graph")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PRED_DIR = os.path.join(PROJECT_ROOT, "predictions")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Hyperparameters
HIDDEN_DIM = 64
HEADS_1 = 8
HEADS_2 = 4
DROPOUT = 0.3
LR = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 100
PATIENCE = 10


def compute_class_weights(y):
    """Compute class weights inversely proportional to frequency."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, data, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, data.edge_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    preds = logits.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(data.edge_y.cpu().numpy(), preds)

    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, criterion, device):
    """Evaluate model on a data split."""
    model.eval()
    data = data.to(device)

    logits = model(data)
    loss = criterion(logits, data.edge_y)

    preds = logits.argmax(dim=1).cpu().numpy()
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_true = data.edge_y.cpu().numpy()
    acc = accuracy_score(y_true, preds)

    return loss.item(), acc, preds, probs, y_true


def main():
    print("\n" + "#" * 60)
    print("#  GAT TRAINING PIPELINE")
    print("#" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("  CUDNN Benchmark enabled")

    # Load graph data
    print("\n" + "=" * 60)
    print("LOADING GRAPH DATA")
    print("=" * 60)

    train_data = torch.load(os.path.join(GRAPH_DIR, "train_data.pt"),
                            weights_only=False).to(device)
    test_data = torch.load(os.path.join(GRAPH_DIR, "test_data.pt"),
                           weights_only=False).to(device)

    print(f"  Train: {train_data.num_nodes} nodes, {train_data.num_edges} edges")
    print(f"  Test:  {test_data.num_nodes} nodes, {test_data.num_edges} edges")

    num_node_features = train_data.x.shape[1]
    num_edge_features = train_data.edge_attr.shape[1]
    num_classes = train_data.num_classes

    print(f"  Node features: {num_node_features}")
    print(f"  Edge features: {num_edge_features}")
    print(f"  Classes: {num_classes}")

    # Build model
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    model = build_gat_model(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        heads_1=HEADS_1,
        heads_2=HEADS_2,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"\n{model}")

    # Loss function with class weights
    class_weights = compute_class_weights(train_data.edge_y.cpu().numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\n  Class weights: {class_weights.cpu().numpy()}")

    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ---- Training loop ----
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_data, optimizer, criterion, device
        )

        # Evaluate on test
        test_loss, test_acc, _, _, _ = evaluate(
            model, test_data, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # Scheduler step
        scheduler.step(test_acc)

        # Logging
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Test  Loss: {test_loss:.4f}  Acc: {test_acc:.4f}", end="")

        # Early stopping check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            os.makedirs(MODEL_DIR, exist_ok=True)
            best_model_path = os.path.join(MODEL_DIR, "gat_best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print("  * best", end="")
        else:
            patience_counter += 1

        print()

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best test accuracy: {best_test_acc:.4f} (epoch {best_epoch})")

    # ---- Final evaluation with best model ----
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (best model)")
    print("=" * 60)

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc, y_pred, y_prob, y_true = evaluate(
        model, test_data, criterion, device
    )

    # Load label encoder for class names
    le_path = os.path.join(DATA_DIR, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        digits=4,
    ))

    # Save predictions
    os.makedirs(PRED_DIR, exist_ok=True)
    np.save(os.path.join(PRED_DIR, "gat_y_pred.npy"), y_pred)
    np.save(os.path.join(PRED_DIR, "gat_y_prob.npy"), y_prob)
    print(f"  Predictions saved to {PRED_DIR}")

    # Save training history
    with open(os.path.join(MODEL_DIR, "gat_history.pkl"), "wb") as f:
        pickle.dump(history, f)
    print(f"  Training history saved to {MODEL_DIR}")

    print("\n" + "#" * 60)
    print("#  GAT TRAINING COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()
