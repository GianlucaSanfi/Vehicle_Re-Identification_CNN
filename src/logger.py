import csv
import os
import matplotlib.pyplot as plt


class TrainLogger:
    """
    Logs training/validation metrics and produces plots.
    Keeps everything organized in a single experiment folder.
    """
    def __init__(self, save_dir, dataset, backbone, epochs, attention):
        self.save_dir = save_dir
        self.dataset = dataset
        self.backbone = backbone
        self.epochs = epochs
        self.attention = attention
        os.makedirs(save_dir, exist_ok=True)

        self.log_file = os.path.join(save_dir, "training_log.csv")

        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "rank1",
                "map"
            ])

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "rank1": [],
            "map": []
        }

    def log(self, epoch, train_loss, val_loss, rank1, map_score):
        # Save into CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                rank1,
                map_score
            ])

        # Keep in memory
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["rank1"].append(rank1)
        self.history["map"].append(map_score)

    def plot(self):
        # LOSS CURVE
        plt.figure(figsize=(8, 5))
        plt.plot(self.history["epoch"], self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["epoch"], self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        tit = "Training & Validation Loss {dataset} {backbone}"
        if attention:
            tit += " with attention"
        plt.title(tit)
        plt.legend()
        plt.grid(True)
        name_pt = "loss_curve_{dataset}_{backbone}_{epochs}ep"
        if attention:
            name_pt += "_attention"
        name_pt += ".png"
        plt.savefig(os.path.join(self.save_dir, name_pt))
        plt.close()

        # RANK-1 AND mAP CURVE
        plt.figure(figsize=(8, 5))
        plt.plot(self.history["epoch"], self.history["rank1"], label="Rank-1")
        plt.plot(self.history["epoch"], self.history["map"], label="mAP")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        tit = "Evaluation Metrics {dataset} {backbone}"
        if attention:
            tit += " with attention"
        plt.title(tit)
        plt.legend()
        plt.grid(True)
        name_pt = "metrics_curve_{dataset}_{backbone}_{epochs}ep"
        if attention:
            name_pt += "_attention"
        name_pt += ".png"
        plt.savefig(os.path.join(self.save_dir, name_pt))
        plt.close()
