import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from ..model.MLPClassifier import MLPClassifier
from ..model.mlp import train_mlp




def run_random_search(X, y, param_grid, k=3, epochs=30, batch_size=256, patience=5, device='cpu'):
    best_params, best_avg_f1, best_model = None, -1, None
    results = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for params in param_grid:
        hidden_dims = params['hidden_dims']
        lr = params['lr']
        dropout = params['dropout']
        threshold = 0.5
        fold_f1_scores = []
        best_f1_for_params = -1
        print(f"\nTesting params: hidden_dims={hidden_dims}, lr={lr}, dropout={dropout}, threshold={threshold}")

        model = MLPClassifier(input_dim=X.shape[1], hidden_dims=hidden_dims, dropout_rate=dropout)

        fold_accuracies, fold_precisions, fold_recalls = [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            model.reset_weights()
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

            model, logs = train_mlp(model, train_loader, val_loader, threshold=threshold,
                                   epochs=epochs, lr=lr, patience=patience, device=device, return_logs=True)

            plt.figure(figsize=(8, 5))
            plt.plot(logs['train_loss'], label='Train Loss')
            plt.plot(logs['val_loss'], label='Val Loss')
            plt.plot(logs['val_f1'], label='Val F1')
            plt.xlabel('Epoch')
            plt.ylabel('Loss / F1')
            plt.title(f'Fold {fold} Training Curve')
            plt.legend()
            plt.tight_layout()
            plt.show()


            model.eval()
            y_preds = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    preds = (torch.sigmoid(model(X_batch)) >= threshold).int()
                    y_preds.append(preds.cpu())

            y_val_np = y_val.cpu().numpy()
            y_pred_np = torch.cat(y_preds).numpy()
            f1 = f1_score(y_val_np, y_pred_np)
            accuracy = accuracy_score(y_val_np, y_pred_np)
            precision = precision_score(y_val_np, y_pred_np, zero_division=0)
            recall = recall_score(y_val_np, y_pred_np, zero_division=0)

            print(f"Fold {fold} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            fold_f1_scores.append(f1)
            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)

            best_f1_for_params = max(best_f1_for_params, f1)

        avg_f1 = np.mean(fold_f1_scores)
        avg_accuracy = np.mean(fold_accuracies)
        avg_precision = np.mean(fold_precisions)
        avg_recall = np.mean(fold_recalls)

        results.append((params, avg_f1))

        print(f"🔹 Avg Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_params = params
            best_model = MLPClassifier(input_dim=X.shape[1], hidden_dims=hidden_dims, dropout_rate=dropout)
            best_model.load_state_dict(model.state_dict())

    return best_params, results

# Generate Random Param Grid
def generate_random_param_grid(n_samples=6):
    hidden_dims_choices = [[128, 64], [512, 256], [256, 128, 64]]
    lr_choices = [0.0001, 0.0005, 0.001]
    dropout_choices=[0.1,0.2,0.3]
    grid = []
    for _ in range(n_samples):
        grid.append({
            'hidden_dims': random.choice(hidden_dims_choices),
            'lr': random.choice(lr_choices),
            'dropout': random.choice(dropout_choices),
        })
    print("Generated Parameter Grid:")
    for i, params in enumerate(grid, 1):
        print(f"{i}. hidden_dims: {params['hidden_dims']}, lr: {params['lr']}, dropout: {params['dropout']}")
    
    return grid
