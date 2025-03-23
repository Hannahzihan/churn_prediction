import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from src.inference.predict import predict_proba_torch

def plot_precision_recall(y_true, model_probs_dict):
    plt.figure(figsize=(6, 5))
    for name, probs in model_probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, model_probs_dict):
    plt.figure(figsize=(6, 5))
    for name, probs in model_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def best_threshold(model, X_tensor, y_true, device='cpu', plot=True):
    X_tensor = X_tensor.to(device)
    probs = predict_proba_torch(model, X_tensor)

    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
        plt.axvline(best_threshold, linestyle='--', color='red', label=f'Best Threshold = {best_threshold:.2f}')
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs. Threshold")
        plt.legend()
        plt.grid(True)
        plt.show()

    return best_threshold, best_f1
