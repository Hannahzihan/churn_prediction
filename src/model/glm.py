from ..data.resample import data_resampling
from sklearn.linear_model import LogisticRegression

def train_glm(X_train, y_train, downsample_ratio=1.0):
    X_train_resampled, y_train_resampled = data_resampling(X_train, y_train, downsample_ratio)

    glm_model = LogisticRegression(max_iter=1000, random_state=42)
    glm_model.fit(X_train_resampled, y_train_resampled)

    return glm_model
