from sklearn.metrics import classification_report, roc_auc_score
from ..inference.predict import predict_proba_torch

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} Evaluation Report:")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return y_pred, y_prob

def evaluate_torch_model(model, X_tensor, y_true, threshold=0.5, model_name="Model", device=None):
    y_prob = predict_proba_torch(model, X_tensor)
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{model_name} Evaluation Report:")
    print(classification_report(y_true, y_pred))
    print(f"{model_name} AUC: {roc_auc_score(y_true, y_prob):.4f}")

    return y_pred, y_prob
