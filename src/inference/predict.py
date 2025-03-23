import torch

def predict_proba_torch(model, X_tensor):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_tensor)).cpu().numpy().flatten()
    return probs