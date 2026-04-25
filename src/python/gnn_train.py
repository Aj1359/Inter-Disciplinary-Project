#!/usr/bin/env python3
"""
GNN Training using pre-computed training data from C++ gen_training_data.

Steps:
  1. Load pre-computed graph data (features + BC) from CSV files
  2. Load edge lists for graph structure
  3. Train DrBC-style GNN with pairwise + listwise ranking loss
  4. Validate on held-out graphs
  5. Save best checkpoint
"""
import argparse
import glob
import math
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnn_model import DrBCModel, NODE_FEAT_DIM


def load_precomputed_graph(csv_path, edge_path):
    """Load a graph from pre-computed CSV + edge list."""
    # Load node features and BC
    bc_values = []
    features = []

    with open(csv_path, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            node_id = int(parts[0])
            bc = float(parts[1])
            feats = [float(x) for x in parts[2:10]]
            bc_values.append(bc)
            features.append(feats)

    n = len(bc_values)
    if n == 0:
        return None

    # Load edge list
    adj = [[] for _ in range(n)]
    with open(edge_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if 0 <= u < n and 0 <= v < n:
                    adj[u].append(v)
                    adj[v].append(u)

    deg = [len(adj[i]) for i in range(n)]
    x = torch.tensor(features, dtype=torch.float32)

    return {
        "adj": adj,
        "deg": deg,
        "n": n,
        "bc": bc_values,
        "x": x,
    }


def load_all_graphs(data_dir):
    """Load all pre-computed graphs from a directory."""
    graphs = []
    csv_files = sorted(glob.glob(os.path.join(data_dir, "graph_*[0-9].csv")))

    for csv_path in csv_files:
        # Derive edge file path
        base = csv_path.replace(".csv", "")
        edge_path = base + "_edges.txt"
        if not os.path.exists(edge_path):
            continue

        g = load_precomputed_graph(csv_path, edge_path)
        if g is not None and g["n"] >= 20:
            graphs.append(g)

    return graphs


def sample_pairs(bc_values, num_pairs, rng):
    """Sample pairs of nodes with different BC values."""
    n = len(bc_values)
    nonzero = [i for i in range(n) if bc_values[i] > 0]
    if len(nonzero) < 2:
        return []

    pairs = []
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 5:
        attempts += 1
        i = rng.choice(nonzero)
        j = rng.choice(nonzero)
        if i != j and bc_values[i] != bc_values[j]:
            pairs.append((i, j))
    return pairs


def pairwise_ranking_loss(scores, bc_values, pairs, margin=0.3):
    """BPR-style pairwise ranking loss."""
    if not pairs:
        return torch.tensor(0.0, requires_grad=True)

    loss = torch.tensor(0.0, device=scores.device, requires_grad=False)
    count = 0

    for i, j in pairs:
        if bc_values[i] > bc_values[j]:
            diff = scores[i] - scores[j]
        elif bc_values[j] > bc_values[i]:
            diff = scores[j] - scores[i]
        else:
            continue

        loss = loss + F.relu(margin - diff)
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True)
    return loss / count


def listwise_loss(scores, bc_values, temperature=1.0):
    """ListNet listwise ranking loss."""
    bc_tensor = torch.tensor(bc_values, dtype=torch.float32, device=scores.device)
    mask = bc_tensor > 0
    if mask.sum() < 2:
        return torch.tensor(0.0, requires_grad=True)

    bc_sub = bc_tensor[mask]
    sc_sub = scores[mask]

    true_dist = F.softmax(torch.log(bc_sub + 1e-10) / temperature, dim=0)
    pred_dist = F.log_softmax(sc_sub / temperature, dim=0)

    loss = -(true_dist * pred_dist).sum()
    return loss


def train_epoch(model, optimizer, graphs, rng, device, pairs_per_graph=300):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_graphs = 0

    indices = list(range(len(graphs)))
    rng.shuffle(indices)

    for idx in indices:
        g = graphs[idx]
        x = g["x"].to(device)
        scores = model(x, g["adj"], g["deg"])

        pairs = sample_pairs(g["bc"], pairs_per_graph, rng)
        if not pairs:
            continue

        L_pair = pairwise_ranking_loss(scores, g["bc"], pairs)
        L_list = listwise_loss(scores, g["bc"])
        loss = 0.5 * L_pair + 0.5 * L_list

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_graphs += 1

    return total_loss / max(1, num_graphs)


def evaluate(model, graphs, rng, device, num_pairs=300):
    """Evaluate ordering accuracy."""
    model.eval()
    total_correct = 0
    total_pairs = 0

    with torch.no_grad():
        for g in graphs:
            x = g["x"].to(device)
            scores = model(x, g["adj"], g["deg"])
            score_np = scores.cpu().numpy()

            pairs = sample_pairs(g["bc"], num_pairs, rng)
            for i, j in pairs:
                pred = score_np[i] > score_np[j]
                actual = g["bc"][i] > g["bc"][j]
                if pred == actual:
                    total_correct += 1
                total_pairs += 1

    return total_correct / max(1, total_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="training_data")
    parser.add_argument("--val-dir", default="val_data")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pairs-per-graph", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="gnn_model.pt")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # Load data
    print("Loading training graphs...")
    train_graphs = load_all_graphs(args.train_dir)
    print(f"  Loaded {len(train_graphs)} training graphs")

    print("Loading validation graphs...")
    val_graphs = load_all_graphs(args.val_dir)
    print(f"  Loaded {len(val_graphs)} validation graphs")

    if not train_graphs:
        raise SystemExit("No training graphs found! Run gen_training_data first.")

    # Create model
    model = DrBCModel(
        in_dim=NODE_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Training
    best_val_acc = 0.0
    best_epoch = 0
    patience = 12
    patience_counter = 0

    print(f"\n{'Epoch':>6} {'Loss':>10} {'TrainAcc':>10} {'ValAcc':>10} {'LR':>10} {'Time':>7}")
    print("=" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        avg_loss = train_epoch(
            model, optimizer, train_graphs, rng, device, args.pairs_per_graph)

        train_acc = evaluate(model, train_graphs[:10], rng, device, 200)
        val_acc = evaluate(model, val_graphs, rng, device, 300)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "in_dim": NODE_FEAT_DIM,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
                "val_acc": val_acc,
                "epoch": epoch,
            }, args.model_out)
            marker = " *"
        else:
            patience_counter += 1

        print(f"{epoch:>6} {avg_loss:>10.4f} {train_acc*100:>9.2f}% "
              f"{val_acc*100:>9.2f}% {lr:>10.6f} {elapsed:>6.1f}s{marker}")

        scheduler.step(val_acc)

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*55}")
    print(f"Best: epoch {best_epoch}, val_acc={best_val_acc*100:.2f}%")
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
