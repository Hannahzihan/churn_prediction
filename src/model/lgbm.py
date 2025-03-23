import lightgbm as lgb
from ..data.resample import data_resampling

def train_lgbm(X_train, y_train, best_params=None, downsample_ratio=1.0):
    X_train_resampled, y_train_resampled = data_resampling(X_train, y_train, downsample_ratio)

    if best_params is None:
        best_params = {}

    lgbm_model = lgb.LGBMClassifier(objective='binary', random_state=42, **best_params)
    lgbm_model.fit(X_train_resampled, y_train_resampled)

    return lgbm_model
