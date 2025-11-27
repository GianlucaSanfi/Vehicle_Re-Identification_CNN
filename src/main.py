# main.py
import argparse
import torch
from globals import DEVICE, EPOCHS, LR, BATCH_P, BATCH_K, MARGIN
from data_loader import load_dataset, build_train_loader, build_test_loader
from network import ReIDModel
from train import train_one_epoch
from evaluate import extract_features, compute_distance_matrix, evaluate_metrics
from utils import BatchHardTripletLoss, set_seed
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

set_seed(42)

def main():
    parser = argparse.ArgumentParser(description="Vehicle Re-ID Training & Evaluation")
    parser.add_argument("--dataset", type=str, default="VRU", help="Dataset to use (VRU/VeRi776)")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    args = parser.parse_args()

    # ------------------------
    # Load Dataset
    # ------------------------
    print(f"Loading dataset: {args.dataset}")
    train_samples, test_samples = load_dataset(args.dataset)
    num_pids = len(set(pid for _, pid in train_samples))

    # ------------------------
    # Train Models
    # ------------------------
    if args.train:
        for backbone in ["resnet18", "resnet50"]:
            print(f"\nTraining {backbone}...")
            model = ReIDModel(num_classes=num_pids, backbone=backbone).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            ce_loss = nn.CrossEntropyLoss()
            tri_loss = BatchHardTripletLoss(margin=MARGIN)

            train_loader = build_train_loader(train_samples, batch_p=BATCH_P, batch_k=BATCH_K)

            for epoch in range(EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, ce_loss, tri_loss, epoch)
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}")

            # Save checkpoint
            model_path = f"{args.dataset}_{backbone}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved {backbone} model to {model_path}")

    # ------------------------
    # Evaluate Models
    # ------------------------
    if args.evaluate:
        for backbone in ["resnet18", "resnet50"]:
            print(f"\nEvaluating {backbone}...")
            model = ReIDModel(num_classes=num_pids, backbone=backbone).to(DEVICE)
            model_path = f"{args.dataset}_{backbone}.pth"
            if not os.path.exists(model_path):
                print(f"Model {model_path} not found. Train first!")
                continue
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            test_loader = build_test_loader(test_samples, batch_size=64)
            features, pids = extract_features(model, test_loader)

            # Compute Cosine Distance
            dist_cos = compute_distance_matrix(features, features, metric="cosine")
            mAP_cos, cmc_cos = evaluate_metrics(dist_cos, pids)
            print(f"[{backbone}] Cosine Distance: mAP={mAP_cos:.4f}, Rank-1={cmc_cos[0]:.4f}")

            # Compute Euclidean Distance
            dist_euc = compute_distance_matrix(features, features, metric="euclidean")
            mAP_euc, cmc_euc = evaluate_metrics(dist_euc, pids)
            print(f"[{backbone}] Euclidean Distance: mAP={mAP_euc:.4f}, Rank-1={cmc_euc[0]:.4f}")

if __name__ == "__main__":
    main()
