
from imblearn.over_sampling import SMOTE
import pandas as pd

def data_resampling(X, y, downsample_ratio=1.0):
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    df = pd.DataFrame(X_res, columns=X.columns if isinstance(X, pd.DataFrame) else None)

    df['label'] = y_res
    majority = df[df['label'] == 0]
    minority = df[df['label'] == 1]
    majority_down = majority.sample(n=int(len(minority) * downsample_ratio), random_state=42)
    df_resampled = pd.concat([majority_down, minority], axis=0).sample(frac=1, random_state=42)

    X_final = df_resampled.drop(columns='label')
    y_final = df_resampled['label']
    return X_final, y_final

