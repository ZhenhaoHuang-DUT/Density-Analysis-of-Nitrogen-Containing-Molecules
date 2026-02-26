"""
ALIGNN density prediction training script - with visualization features
Integrated:
 - Huber-style loss (nn.SmoothL1Loss)
 - Gradient clipping (torch.nn.utils.clip_grad_norm_)
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Import ALIGNN model
try:
    from ALIGNN_Model import ALIGNN
except ImportError:
    print("Error: alignn_model.py not found")
    exit(1)


# ==================== Configuration ====================
class Config:
    # Data parameters
    data_path = "TrainData/alignn_graph_dataset.pt"  # Dataset path
    batch_size = 64  # Batch size
    train_ratio = 0.8  # Training set ratio
    val_ratio = 0.1  # Validation set ratio
    test_ratio = 0.1  # Test set ratio

    # Model parameters
    node_dim = 9  # Atom feature dimension
    edge_dim = 6  # Edge feature dimension
    line_edge_dim = 5  # Line graph edge feature dimension
    global_dim = 6  # Global feature dimension
    hidden_dim = 256  # Hidden layer dimension
    num_layers = 4  # Number of ALIGNN layers
    dropout = 0.1  # Dropout rate

    # Training parameters
    epochs = 100  # Number of epochs
    lr = 3e-4  # Learning rate
    weight_decay = 1e-5  # Weight decay
    patience = 20  # Early stopping patience
    min_delta = 1e-4  # Minimum improvement threshold

    # Huber / SmoothL1 parameter (beta)
    huber_beta = 0.1  # Smaller beta is closer to MAE, behaves like MSE for small errors (tunable)

    # Gradient clipping
    max_grad_norm = 5.0  # Threshold for clip_grad_norm_

    # Device settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42  # Random seed
    num_workers = 0
    # Output settings
    output_dir = "Training_Results"  # Output directory
    save_model = True  # Whether to save model
    plot_frequency = 5  # Plot frequency (every N epochs)


# ==================== Dataset Class ====================
class MoleculeDataset(Dataset):
    """Molecular dataset"""

    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    """Custom collate function"""
    return Batch.from_data_list(batch)


# ==================== Trainer Class ====================
class ALIGNNTrainer:
    """ALIGNN trainer"""

    def __init__(self, config):
        self.config = config
        self.setup_environment()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_visualization()

    def setup_environment(self):
        """Set up training environment"""
        # Set random seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        print(f"Using device: {self.config.device}")
        print(f"Output directory: {self.config.output_dir}")

    def setup_data(self):
        """Set up data loaders"""
        print(f"Loading data: {self.config.data_path}")

        # Load data
        data_list = torch.load(self.config.data_path)
        print(f"Total samples: {len(data_list)}")

        # Create dataset
        dataset = MoleculeDataset(data_list)

        # Split dataset
        n_total = len(dataset)
        n_train = int(self.config.train_ratio * n_total)
        n_val = int(self.config.val_ratio * n_total)
        n_test = n_total - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers
        )

        print(f"Train set: {len(train_dataset)}, Validation set: {len(val_dataset)}, Test set: {len(test_dataset)}")

        # Compute density statistics for the dataset
        self.compute_density_statistics(data_list)

    def compute_density_statistics(self, data_list):
        """Compute density statistics"""
        # Handle possible shape (1,) for data.y by stacking and squeezing
        densities = torch.stack([d.y.view(-1) for d in data_list]).numpy().flatten()

        self.density_stats = {
            'mean': float(np.mean(densities)),
            'std': float(np.std(densities)),
            'min': float(np.min(densities)),
            'max': float(np.max(densities)),
            'median': float(np.median(densities)),
        }

        print("\nDensity statistics:")
        print(f"  Mean: {self.density_stats['mean']:.4f}")
        print(f"  Std: {self.density_stats['std']:.4f}")
        print(f"  Range: [{self.density_stats['min']:.4f}, {self.density_stats['max']:.4f}]")
        print(f"  Median: {self.density_stats['median']:.4f}")
    def setup_model(self):
        """Initialize model"""
        print("\nInitializing ALIGNN model...")

        self.model = ALIGNN(
            node_dim=self.config.node_dim,
            edge_dim=self.config.edge_dim,
            line_edge_dim=self.config.line_edge_dim,
            global_dim=self.config.global_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.config.device)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model architecture: {self.model.__class__.__name__}")

    def setup_optimizer(self):
        """Set up optimizer and loss function"""
        # Use SmoothL1Loss (PyTorch's Huber-style implementation)
        self.criterion = nn.SmoothL1Loss(beta=self.config.huber_beta)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

    def setup_visualization(self):
        """Set up visualization"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': [],
            'lr': []
        }

        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            batch = batch.to(self.config.device)

            self.optimizer.zero_grad()

            # Forward pass
            density_pred = self.model(batch)  # (batch_size, 1)
            targets = batch.y
            if targets.dim() == 1:
                targets = targets.view(-1, 1)
            targets = targets.to(dtype=density_pred.dtype, device=density_pred.device)

            loss = self.criterion(density_pred, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping (use max_grad_norm from config)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def evaluate(self, dataloader, phase='Validation'):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_ids = []

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=phase, leave=False)
            for batch in progress_bar:
                batch = batch.to(self.config.device)

                density_pred = self.model(batch)
                targets = batch.y
                if targets.dim() == 1:
                    targets = targets.view(-1, 1)

                # Ensure dtype/device match
                targets = targets.to(dtype=density_pred.dtype, device=density_pred.device)

                loss = self.criterion(density_pred, targets)
                total_loss += loss.item()

                all_preds.extend(density_pred.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

                # Collect molecule IDs if present
                if hasattr(batch, 'mol_id'):
                    try:
                        all_ids.extend(batch.mol_id.cpu().numpy().flatten())
                    except Exception:
                        pass

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Compute metrics
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)

        # Compute mean absolute percentage error
        mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100

        metrics = {
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else float('nan'),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'preds': all_preds,
            'targets': all_targets,
            'ids': all_ids if all_ids else None
        }

        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            # Save a clean config (only basic types)
            'config': {
                'node_dim': self.config.node_dim,
                'edge_dim': self.config.edge_dim,
                'line_edge_dim': self.config.line_edge_dim,
                'global_dim': self.config.global_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'dropout': self.config.dropout,
                'lr': self.config.lr,
                'weight_decay': self.config.weight_decay,
                'huber_beta': self.config.huber_beta,
                'max_grad_norm': self.config.max_grad_norm
            }
        }

        if is_best:
            torch.save(checkpoint, os.path.join(self.config.output_dir, 'best_model.pth'))
            print(f"Saved best model at epoch {epoch}")
        else:
            torch.save(checkpoint, os.path.join(self.config.output_dir, f'checkpoint_epoch_{epoch}.pth'))

    def plot_training_history(self, epoch):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color=self.colors[0], linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', color=self.colors[1], linewidth=2)
        axes[0, 0].axhline(y=self.best_val_loss, color='red', linestyle='--', alpha=0.5,
                           label=f'Best Loss: {self.best_val_loss:.4f}')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. R² scores
        axes[0, 1].plot(self.history['val_r2'], label='Validation R²', color=self.colors[2], linewidth=2)
        best_r2 = max(self.history['val_r2']) if self.history['val_r2'] else 0
        axes[0, 1].axhline(y=best_r2, color='green', linestyle='--', alpha=0.5, label=f'Best R²: {best_r2:.4f}')
        axes[0, 1].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('Validation R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. MAE and RMSE
        epochs = range(1, len(self.history['val_mae']) + 1)
        axes[0, 2].plot(epochs, self.history['val_mae'], label='MAE', color=self.colors[3], linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_rmse'], label='RMSE', color=self.colors[4], linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Error')
        axes[0, 2].set_title('Validation Error Metrics')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Learning rate changes
        if self.history['lr']:
            axes[1, 0].plot(self.history['lr'], color=self.colors[0], linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Loss distribution (last 5 epochs)
        if len(self.history['val_loss']) >= 5:
            recent_losses = self.history['val_loss'][-5:]
            axes[1, 1].bar(range(len(recent_losses)), recent_losses, color=self.colors[1])
            axes[1, 1].set_xlabel('Last 5 Epochs')
            axes[1, 1].set_ylabel('Validation Loss')
            axes[1, 1].set_title('Validation Loss (Last 5 Epochs)')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        # 6. Metrics comparison
        metrics_data = {
            'MAE': self.history['val_mae'][-1] if self.history['val_mae'] else 0,
            'RMSE': self.history['val_rmse'][-1] if self.history['val_rmse'] else 0,
            'R²': self.history['val_r2'][-1] if self.history['val_r2'] else 0,
        }
        axes[1, 2].bar(metrics_data.keys(), metrics_data.values(), color=self.colors[:3])
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_title('Current Validation Metrics')
        for i, (key, value) in enumerate(metrics_data.items()):
            axes[1, 2].text(i, value, f'{value:.4f}', ha='center', va='bottom')
        axes[1, 2].grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'ALIGNN Training History (Epoch {epoch})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, f'training_history_epoch_{epoch}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()
    def plot_predictions(self, preds, targets, phase='Validation', epoch=None):
        """Plot prediction results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Predictions vs True values scatter plot
        axes[0, 0].scatter(targets, preds, alpha=0.5, s=10, color=self.colors[0])

        # Draw diagonal line
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Ideal Prediction')

        # Calculate regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(targets, preds)
        regression_line = slope * np.array([min_val, max_val]) + intercept
        axes[0, 0].plot([min_val, max_val], regression_line, 'g-', alpha=0.8,
                        label=f'Regression: y={slope:.3f}x+{intercept:.3f}\nR²={r_value ** 2:.4f}')

        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title(f'{phase} Set: Predictions vs True Values (R²={r_value ** 2:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residual plot
        residuals = targets - preds
        axes[0, 1].scatter(preds, residuals, alpha=0.5, s=10, color=self.colors[1])
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Predictions')
        axes[0, 1].set_ylabel('Residuals (True - Pred)')
        axes[0, 1].set_title(f'{phase} Set: Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Error distribution histogram
        axes[1, 0].hist(residuals, bins=50, color=self.colors[2], alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'{phase} Set: Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        axes[1, 0].text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
                        transform=axes[1, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].get_lines()[0].set_marker('.')
        axes[1, 1].get_lines()[0].set_markersize(5)
        axes[1, 1].get_lines()[0].set_markerfacecolor(self.colors[3])
        axes[1, 1].get_lines()[0].set_markeredgecolor(self.colors[3])
        axes[1, 1].get_lines()[1].set_color('red')
        axes[1, 1].get_lines()[1].set_linewidth(2)
        axes[1, 1].set_title(f'{phase} Set: Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)

        title = f'ALIGNN {phase} Set Predictions' + (f' (Epoch {epoch})' if epoch else '')
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        filename = f'{phase.lower()}_predictions'
        if epoch:
            filename += f'_epoch_{epoch}'
        plt.savefig(os.path.join(self.config.output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_detailed_report(self, train_metrics, val_metrics, test_metrics):
        """Create detailed report"""
        report = {
            'model_info': {
                'model_name': 'ALIGNN',
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            },
            'training_config': vars(self.config),
            'data_statistics': {
                'density_stats': self.density_stats,
                'dataset_sizes': {
                    'train_set': len(self.train_loader.dataset),
                    'validation_set': len(self.val_loader.dataset),
                    'test_set': len(self.test_loader.dataset),
                }
            },
            'training_history': {
                'best_validation_loss': self.best_val_loss,
                'total_epochs_trained': len(self.history['train_loss']),
                'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            },
            'final_performance': {
                'train_set': {
                    'loss': train_metrics['loss'],
                    'MAE': train_metrics['mae'],
                    'RMSE': train_metrics['rmse'],
                    'R2': train_metrics['r2'],
                    'MAPE': train_metrics['mape'],
                },
                'validation_set': {
                    'loss': val_metrics['loss'],
                    'MAE': val_metrics['mae'],
                    'RMSE': val_metrics['rmse'],
                    'R2': val_metrics['r2'],
                    'MAPE': val_metrics['mape'],
                },
                'test_set': {
                    'loss': test_metrics['loss'],
                    'MAE': test_metrics['mae'],
                    'RMSE': test_metrics['rmse'],
                    'R2': test_metrics['r2'],
                    'MAPE': test_metrics['mape'],
                }
            }
        }

        # Save JSON report
        report_path = os.path.join(self.config.output_dir, 'training_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Create text summary
        summary_path = os.path.join(self.config.output_dir, 'training_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ALIGNN Density Prediction Training Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write("Model configuration:\n")
            f.write("-" * 40 + "\n")
            for key, value in vars(self.config).items():
                if key != 'device':  # skip device object
                    f.write(f"  {key}: {value}\n")

            f.write("\nData statistics:\n")
            f.write("-" * 40 + "\n")
            for key, value in self.density_stats.items():
                f.write(f"  density_{key}: {value:.4f}\n")

            f.write(f"\nDataset sizes:\n")
            f.write(f"  Train set: {len(self.train_loader.dataset)}\n")
            f.write(f"  Validation set: {len(self.val_loader.dataset)}\n")
            f.write(f"  Test set: {len(self.test_loader.dataset)}\n")

            f.write("\nPerformance metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Best validation loss: {self.best_val_loss:.6f}\n")
            f.write(f"  Total epochs trained: {len(self.history['train_loss'])}\n")

            f.write("\nTest set results:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Loss: {test_metrics['loss']:.6f}\n")
            f.write(f"  MAE: {test_metrics['mae']:.6f}\n")
            f.write(f"  RMSE: {test_metrics['rmse']:.6f}\n")
            f.write(f"  R2: {test_metrics['r2']:.6f}\n")
            f.write(f"  MAPE: {test_metrics['mape']:.2f}%\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"Detailed report saved to: {report_path}")
        print(f"Training summary saved to: {summary_path}")

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("Starting ALIGNN model training")
        print("=" * 60)

        patience_counter = 0
        best_r2 = -float('inf')

        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-" * 40)

            # Training
            train_loss = self.train_epoch()

            # Validation
            val_metrics = self.evaluate(self.val_loader, 'Validation')

            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_r2'].append(val_metrics['r2'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Print progress
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Validation Loss: {val_metrics['loss']:.6f}")
            print(f"Validation MAE: {val_metrics['mae']:.6f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.6f}")
            print(f"Validation R²: {val_metrics['r2']:.6f}")
            print(f"Validation MAPE: {val_metrics['mape']:.2f}%")

            # Check for best model
            if val_metrics['loss'] < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0

                # Save best model
                if self.config.save_model:
                    self.save_checkpoint(epoch, is_best=True)

                # Update best R2
                if val_metrics['r2'] > best_r2:
                    best_r2 = val_metrics['r2']

                print(f"🎯 New best model found! Loss: {val_metrics['loss']:.6f}, R²: {val_metrics['r2']:.6f}")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{self.config.patience}")

            # Periodic plotting
            if epoch % self.config.plot_frequency == 0 or epoch == 1:
                self.plot_training_history(epoch)

                # Plot current validation predictions
                self.plot_predictions(
                    val_metrics['preds'],
                    val_metrics['targets'],
                    phase='Validation',
                    epoch=epoch
                )

            # Save checkpoint periodically
            if self.config.save_model and epoch % 10 == 0:
                self.save_checkpoint(epoch)

            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
                break

        # Training finished
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)

        # Load best model if available
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Best model loaded")

        # Final evaluation
        print("\nFinal evaluation:")
        print("-" * 40)

        # Train set evaluation
        train_metrics = self.evaluate(self.train_loader, 'Train')
        print(f"Train set - Loss: {train_metrics['loss']:.6f}, R2: {train_metrics['r2']:.6f}")

        # Validation set evaluation
        val_metrics = self.evaluate(self.val_loader, 'Validation')
        print(f"Validation set - Loss: {val_metrics['loss']:.6f}, R2: {val_metrics['r2']:.6f}")

        # Test set evaluation
        test_metrics = self.evaluate(self.test_loader, 'Test')
        print(f"Test set - Loss: {test_metrics['loss']:.6f}, R2: {test_metrics['r2']:.6f}")

        # Plot final predictions and history
        self.plot_training_history(len(self.history['train_loss']))
        self.plot_predictions(train_metrics['preds'], train_metrics['targets'], 'Training', 'Final')
        self.plot_predictions(val_metrics['preds'], val_metrics['targets'], 'Validation', 'Final')
        self.plot_predictions(test_metrics['preds'], test_metrics['targets'], 'Test', 'Final')

        # Create comprehensive performance plots
        self.plot_comprehensive_results(train_metrics, val_metrics, test_metrics)

        # Create detailed report
        self.create_detailed_report(train_metrics, val_metrics, test_metrics)

        # Save final model
        if self.config.save_model:
            final_model_path = os.path.join(self.config.output_dir, 'final_model.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {
                    'node_dim': self.config.node_dim,
                    'edge_dim': self.config.edge_dim,
                    'line_edge_dim': self.config.line_edge_dim,
                    'global_dim': self.config.global_dim,
                    'hidden_dim': self.config.hidden_dim,
                    'num_layers': self.config.num_layers,
                    'dropout': self.config.dropout,
                    'lr': self.config.lr,
                    'weight_decay': self.config.weight_decay,
                    'huber_beta': self.config.huber_beta,
                    'max_grad_norm': self.config.max_grad_norm
                },
                'history': self.history,
                'density_stats': self.density_stats,
                'test_metrics': test_metrics
            }, final_model_path)
            print(f"\nFinal model saved to: {final_model_path}")

        return test_metrics['r2']

    def plot_comprehensive_results(self, train_metrics, val_metrics, test_metrics):
        """Plot comprehensive performance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. R² comparison across three datasets
        datasets = ['Train Set', 'Validation Set', 'Test Set']
        r2_values = [train_metrics['r2'], val_metrics['r2'], test_metrics['r2']]

        bars = axes[0, 0].bar(datasets, r2_values, color=self.colors[:3])
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Score Comparison Across Datasets')
        axes[0, 0].set_ylim([0, max(1.0, max(r2_values) * 1.1)])
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Add values on top of bars
        for bar, r2 in zip(bars, r2_values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{r2:.4f}', ha='center', va='bottom')

        # 2. Error comparison across three datasets
        mae_values = [train_metrics['mae'], val_metrics['mae'], test_metrics['mae']]
        rmse_values = [train_metrics['rmse'], val_metrics['rmse'], test_metrics['rmse']]

        x = np.arange(len(datasets))
        width = 0.35

        bars1 = axes[0, 1].bar(x - width / 2, mae_values, width, label='MAE', color=self.colors[0])
        bars2 = axes[0, 1].bar(x + width / 2, rmse_values, width, label='RMSE', color=self.colors[1])

        axes[0, 1].set_ylabel('Error')
        axes[0, 1].set_title('Error Comparison Across Datasets')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(datasets)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. Detailed analysis of test set
        residuals = test_metrics['targets'] - test_metrics['preds']
        absolute_errors = np.abs(residuals)

        # Error distribution box plot
        error_data = [residuals, absolute_errors]
        box = axes[1, 0].boxplot(error_data, patch_artist=True)

        # Set colors
        colors = [self.colors[3], self.colors[4]]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        axes[1, 0].set_xticklabels(['Residuals', 'Absolute Errors'])
        axes[1, 0].set_ylabel('Error Value')
        axes[1, 0].set_title('Test Set Error Distribution')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. Performance metrics radar chart
        metrics_names = ['R²', '1-MAE', '1-RMSE', '1-MAPE']

        # Normalize metrics (convert to 0-1 range, higher is better)
        max_mae = max(mae_values)
        max_rmse = max(rmse_values)
        max_mape = max(train_metrics['mape'], val_metrics['mape'], test_metrics['mape'])

        train_normalized = [
            train_metrics['r2'],  # R² is already 0-1
            1 - (train_metrics['mae'] / max_mae if max_mae > 0 else 0),
            1 - (train_metrics['rmse'] / max_rmse if max_rmse > 0 else 0),
            1 - (train_metrics['mape'] / max_mape if max_mape > 0 else 0)
        ]

        test_normalized = [
            test_metrics['r2'],
            1 - (test_metrics['mae'] / max_mae if max_mae > 0 else 0),
            1 - (test_metrics['rmse'] / max_rmse if max_rmse > 0 else 0),
            1 - (test_metrics['mape'] / max_mape if max_mape > 0 else 0)
        ]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        train_normalized += train_normalized[:1]
        test_normalized += test_normalized[:1]
        angles += angles[:1]

        ax_radar = fig.add_subplot(2, 2, 4, projection='polar')
        ax_radar.plot(angles, train_normalized, 'o-', linewidth=2, label='Train Set', color=self.colors[0])
        ax_radar.fill(angles, train_normalized, alpha=0.25, color=self.colors[0])
        ax_radar.plot(angles, test_normalized, 'o-', linewidth=2, label='Test Set', color=self.colors[2])
        ax_radar.fill(angles, test_normalized, alpha=0.25, color=self.colors[2])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics_names)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Performance Metrics Radar Chart (Normalized)')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax_radar.grid(True)

        plt.suptitle('ALIGNN Model Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ==================== Main Function ====================
def main():
    """Main function"""
    print("=" * 60)
    print("ALIGNN Density Prediction Model Training")
    print("=" * 60)

    # Create config
    config = Config()

    # Print config
    print("\nTraining configuration:")
    print("-" * 40)
    for key, value in vars(config).items():
        if key != 'device':  # skip device object
            print(f"  {key}: {value}")

    # Create trainer and start training
    trainer = ALIGNNTrainer(config)
    final_r2 = trainer.train()

    print(f"\n✅ Training completed! Final Test R²: {final_r2:.4f}")
    print(f"📊 All results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()