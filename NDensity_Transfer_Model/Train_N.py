import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from ALIGNN_Model import ALIGNN
from torch_geometric.loader import DataLoader  # use torch_geometric DataLoader instead of torch.utils.data.DataLoader

CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_path": "TrainData/alignn_graph_dataset_N.pt",
    "batch_size": 32,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "pretrained_ckpt": "../COD_Density_Pretraining/Training_Results/best_model.pth",
    "hidden_dim": 256,
    "adapter_hidden": 64,
    "epochs": 80,
    "lr": 2e-4,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "freeze_backbone_epochs": 15,
    "huber_delta": 0.3,
    "nitrogen_boost": 1.5,
    "output_dir": "results_nitrogen_fullviz",
    "scatter_interval": 5
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NitrogenAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, u_N):
        return self.net(u_N).squeeze(-1)


class Level2DensityModel(nn.Module):
    def __init__(self, backbone, adapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def forward(self, batch):
        pred = self.backbone(batch).squeeze(-1)
        if hasattr(batch, "u_N"):
            pred += self.adapter(batch.u_N.to(pred.device))
        return pred


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, weight):
        err = pred - target
        abs_err = err.abs()
        quad = torch.minimum(abs_err, torch.tensor(self.delta, device=err.device))
        lin = abs_err - quad
        loss = 0.5 * quad ** 2 + self.delta * lin
        return (loss * weight).mean()


def plot_scatter(y_true, y_pred, filename, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--', label='y=x')
    plt.xlabel("True Density")
    plt.ylabel("Predicted Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_hist(residuals, filename, title):
    plt.figure()
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def train():
    set_seed(CONFIG["seed"])
    dataset = torch.load(CONFIG["data_path"])

    # compute basic statistics
    all_y = torch.tensor([data.y.item() for data in dataset])
    mean_y = all_y.mean()
    std_y = all_y.std()

    # define outlier threshold
    threshold = 3.0  # samples beyond mean ± 3*std are considered outliers
    mask = (all_y >= mean_y - threshold * std_y) & (all_y <= mean_y + threshold * std_y)

    # filter dataset
    filtered_dataset = [dataset[i] for i, keep in enumerate(mask) if keep]
    print(f"Original: {len(dataset)} samples, Filtered: {len(filtered_dataset)} samples")
    dataset = filtered_dataset

    n_total = len(dataset)
    n_val = int(n_total * CONFIG["val_ratio"])
    n_test = int(n_total * CONFIG["test_ratio"])
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"])

    # Backbone
    alignn = ALIGNN(
        node_dim=9, edge_dim=6, line_edge_dim=5, global_dim=6,
        hidden_dim=CONFIG["hidden_dim"], num_layers=4, num_ffn_layers=2, dropout=0.1
    )
    ckpt = torch.load(CONFIG["pretrained_ckpt"], map_location=CONFIG["device"])
    alignn.load_state_dict(ckpt['model_state_dict'])
    alignn.to(CONFIG["device"])
    alignn.eval()

    # Adapter + Model
    adapter = NitrogenAdapter(in_dim=dataset[0].u_N.shape[1],
                              hidden_dim=CONFIG["adapter_hidden"])
    model = Level2DensityModel(alignn, adapter).to(CONFIG["device"])

    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])

    criterion = WeightedHuberLoss(CONFIG["huber_delta"])

    best_val = 1e9
    history = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []}

    # CSV record for per-batch detailed information
    csv_records = []

    for epoch in range(CONFIG["epochs"]):
        model.train()

        if epoch < CONFIG["freeze_backbone_epochs"]:
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True

        train_preds, train_targets, train_resid = [], [], []
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(CONFIG["device"])
            pred = model(batch)
            target = batch.y.squeeze()

            train_preds.append(pred.detach().cpu())
            train_targets.append(target.detach().cpu())
            train_resid.append((pred - target).detach().cpu())

            if hasattr(batch, "u_N"):
                n_count = batch.u_N[:, 0]
                weight = 1.0 + CONFIG["nitrogen_boost"] * torch.clamp(n_count / 5.0, max=1.0)
            else:
                weight = torch.ones_like(target)

            loss = criterion(pred, target, weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()

            total_loss += loss.item()

            # save per-sample information
            for i in range(len(target)):
                csv_records.append({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "true": target[i].item(),
                    "pred": pred[i].item(),
                    "residual": (pred[i] - target[i]).item(),
                    "n_count": batch.u_N[i, 0].item() if hasattr(batch, "u_N") else 0
                })

        # epoch-level metrics
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_r2 = r2_score(train_targets, train_preds)

        history["train_loss"].append(total_loss / len(train_loader))
        history["train_r2"].append(train_r2)

        # Validation
        model.eval()
        val_preds, val_targets, val_resid = [], [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(CONFIG["device"])
                pred = model(batch)
                target = batch.y.squeeze()

                val_preds.append(pred.cpu())
                val_targets.append(target.cpu())
                val_resid.append((pred - target).cpu())

                weight = torch.ones_like(target)
                val_loss += criterion(pred, target, weight).item()

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_resid = torch.cat(val_resid)

        val_r2 = r2_score(val_targets, val_preds)

        history["val_loss"].append(val_loss / len(val_loader))
        history["val_r2"].append(val_r2)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {total_loss / len(train_loader):.4f} | "
            f"Val Loss {val_loss / len(val_loader):.4f} | "
            f"Train R2 {train_r2:.4f} | "
            f"Val R2 {val_r2:.4f}"
        )

        # scatter plot every scatter_interval epochs
        if epoch % CONFIG["scatter_interval"] == 0:
            plot_scatter(
                val_targets,
                val_preds,
                os.path.join(CONFIG["output_dir"], f"scatter_epoch{epoch:03d}.png"),
                f"Epoch {epoch}: Val Pred vs True"
            )

            plot_hist(
                val_resid,
                os.path.join(CONFIG["output_dir"], f"residual_epoch{epoch:03d}.png"),
                f"Epoch {epoch}: Residual Distribution"
            )

        # save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(CONFIG["output_dir"], "best_model.pt"))

    # Loss curves
    epochs = range(CONFIG["epochs"])

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG["output_dir"], "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_r2"], label="Train R2")
    plt.plot(epochs, history["val_r2"], label="Val R2")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("Training and Validation R2")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG["output_dir"], "r2_curve.png"))
    plt.close()

    # save CSV records
    pd.DataFrame(csv_records).to_csv(
        os.path.join(CONFIG["output_dir"], "batch_records.csv"),
        index=False
    )


if __name__ == "__main__":
    train()