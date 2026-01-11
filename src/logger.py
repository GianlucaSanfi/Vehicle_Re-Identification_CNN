import csv
import os
import matplotlib.pyplot as plt


class TrainLogger:
    """
    Logs training/validation metrics and produces plots.
    Keeps everything organized in a single experiment folder.
    """
    def __init__(self, save_dir, dataset, backbone, epochs, attention, no_eval):
        self.save_dir = save_dir
        self.dataset = dataset
        self.backbone = backbone
        self.epochs = epochs
        self.attention = attention
        self.no_eval = no_eval
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

        if self.no_eval:
            # LOSS CURVE
            plt.figure(figsize=(8, 5))
            plt.plot(self.history["epoch"], self.history["train_loss"], label="Train Loss")
            plt.plot(self.history["epoch"], self.history["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            tit = "Training & Validation Loss"
            if attention:
                tit += " with attention"
            plt.title(tit)
            plt.legend()
            plt.grid(True)
            name_pt = f"training_summary_{self.dataset}_{self.backbone}_{self.epochs}ep"
            if self.attention:
                name_pt += "_attention"
            name_pt += "_NoEv.png"
            plt.savefig(os.path.join(self.save_dir, name_pt))
            plt.close()


        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # LEFT: LOSS CURVES
            axes[0].plot(self.history["epoch"], self.history["train_loss"], label="Train Loss")
            axes[0].plot(self.history["epoch"], self.history["val_loss"], label="Validation Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training & Validation Loss")
            axes[0].legend()
            axes[0].grid(True)

            # RIGHT: METRICS
            axes[1].plot(self.history["epoch"], self.history["rank1"], label="Rank-1")
            axes[1].plot(self.history["epoch"], self.history["map"], label="mAP")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Score")
            axes[1].set_title("Evaluation Metrics")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            name_pt = f"training_summary_{self.dataset}_{self.backbone}_{self.epochs}ep"
            if self.attention:
                name_pt += "_attention"
            name_pt += ".png"
            plt.savefig(os.path.join(self.save_dir, name_pt))
            plt.close()

