import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

def train_mlp(model, train_loader, val_loader, threshold=0.5, epochs=30, lr=0.001, patience=5,
                device=torch.device("cpu"), return_logs=False, log_path=None):

    model = model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() 
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0
    logs = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(outputs) >= threshold).int()
                y_pred.append(preds.cpu())
                y_true.append(y_val.cpu())

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        val_loss = np.mean(val_losses)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        logs['train_loss'].append(np.mean(train_losses))
        logs['val_loss'].append(val_loss)
        logs['val_f1'].append(f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    model.load_state_dict(best_model_state)
    if log_path:
        pd.DataFrame(logs).to_csv(log_path, index=False)
    return (model, logs) if return_logs else model
