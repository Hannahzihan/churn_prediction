import torch
from src.model.MLPClassifier import MLPClassifier

def mlp_model_loading(model_path, input_size, hidden_sizes=[512, 256], dropout_rate=0.1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = MLPClassifier(input_size, hidden_sizes, dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model