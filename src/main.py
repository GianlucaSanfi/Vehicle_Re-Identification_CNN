# main.py
import argparse
import torch
from globals import DEVICE, EPOCHS, LR, BATCH_P, BATCH_K, MARGIN
from data_loader import load_dataset, build_train_loader, build_test_loader, split_query_gallery
from network import ReIDModel
from train import train_one_epoch
from evaluate import extract_features, compute_distance_matrix, evaluate_metrics
from utils import BatchHardTripletLoss, set_seed
from logger import *
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
    parser.add_argument("--attention", action="store_true", help="implemets attention mechanism (CBAM)")
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
            if args.attention:
                print("Attention ENABLED")
            else:
                print("Attention DISABLED")

            model = ReIDModel(num_classes=num_pids, backbone=backbone, use_attention=args.attention).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            ce_loss = nn.CrossEntropyLoss()
            tri_loss = BatchHardTripletLoss(margin=MARGIN)

            train_loader = build_train_loader(train_samples, batch_p=BATCH_P, batch_k=BATCH_K)

            #path folder for logs and plots
            exp_name = f"{args.dataset}_{backbone}_{EPOCHS}ep"
            if args.attention:
                exp_name += "_attention"

            logger = TrainLogger(os.path.join("logs", exp_name), args.dataset, backbone, EPOCHS, args.attention)
            
            ''' split query and gallery of test-set'''
            #test_loader = build_test_loader(test_samples, batch_size=64)
            query_samples, gallery_samples = split_query_gallery(test_samples)
            query_loader = build_test_loader(query_samples, batch_size=64)
            gallery_loader = build_test_loader(gallery_samples, batch_size=64)

            for epoch in range(EPOCHS):
                loss = train_one_epoch(model, train_loader, optimizer, ce_loss, tri_loss, epoch)
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}")
                
                #for evaluation (query/gallery), logging and plotting 

                query_features, q_pids = extract_features(model, query_loader)
                gallery_features, g_pids = extract_features(model, gallery_loader)
                distmat = compute_distance_matrix(query_features, gallery_features, metric="cosine")
                mAP, cmc = evaluate_metrics(distmat, q_pids, g_pids)
                rank1 = cmc[0]

                lr = optimizer.param_groups[0]["lr"] #optional
                logger.log(epoch=epoch + 1, train_loss=loss, val_loss=0.0, rank1=rank1, map_score=mAP)

            logger.plot()

            # Save checkpoint
            model_path = f"{args.dataset}_{backbone}_{EPOCHS}ep"
            if args.attention:
                model_path += "_attention"
            model_path += ".pth"
            
            torch.save(model.state_dict(), model_path)
            print(f"Saved {backbone} model to {model_path}")

    # ------------------------
    # Evaluate Models
    # ------------------------
    if args.evaluate:
        for backbone in ["resnet18", "resnet50"]:
            print(f"\nEvaluating {backbone}...")
            if args.attention:
                print("Attention ENABLED")
            else:
                print("Attention DISABLED")

            model = ReIDModel(num_classes=num_pids, backbone=backbone, use_attention=args.attention).to(DEVICE)
            model_path = f"{args.dataset}_{backbone}_{EPOCHS}ep"
            if args.attention:
                model_path += "_attention"
            model_path += ".pth"

            if not os.path.exists(model_path):
                print(f"Model {model_path} not found. Train first!")
                continue
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            #test_loader = build_test_loader(test_samples, batch_size=64)
            query_samples, gallery_samples = split_query_gallery(test_samples)
            query_loader = build_test_loader(query_samples, batch_size=64)
            gallery_loader = build_test_loader(gallery_samples, batch_size=64)
            #features, pids = extract_features(model, test_loader)
            query_features, q_pids = extract_features(model, query_loader)
            gallery_features, g_pids = extract_features(model, gallery_loader)
            
            #mAP, cmc = evaluate_metrics(distmat, q_pids, g_pids)

            # Compute Cosine Distance
            dist_cos = compute_distance_matrix(query_features, gallery_features, metric="cosine")
            mAP_cos, cmc_cos = evaluate_metrics(dist_cos, q_pids, g_pids)
            print(f"[{backbone}] Cosine Distance: mAP={mAP_cos:.4f}, Rank-1={cmc_cos[0]:.4f}")

            # Compute Euclidean Distance
            dist_euc = compute_distance_matrix(query_features, gallery_features, metric="euclidean")
            mAP_euc, cmc_euc = evaluate_metrics(dist_euc, q_pids, g_pids)
            print(f"[{backbone}] Euclidean Distance: mAP={mAP_euc:.4f}, Rank-1={cmc_euc[0]:.4f}")

if __name__ == "__main__":
    main()
