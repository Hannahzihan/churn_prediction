
import pandas as pd
import calendar
from sklearn.preprocessing import LabelEncoder, StandardScaler

def data_preprocessing(df, is_train=True, scaler=None, label_encoders=None):
    df = df.copy()

    # Drop columns
    for col in ['msno', 'registration_init_time']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Convert date columns to datetime
    for col in ['last_login_date_previous', 'last_login_date_current']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if all(c in df.columns for c in ['last_login_date_current', 'last_login_date_previous']):
        default_date = pd.to_datetime("1970-01-01")

        # Normal calculation
        df['days_until_month_end'] = ((df['last_login_date_current'] + pd.offsets.MonthEnd(0)) - df['last_login_date_current']).dt.days
        df['last_login_interval'] = (df['last_login_date_current'] - df['last_login_date_previous']).dt.days

        # Masks
        curr_is_default = df['last_login_date_current'] == default_date
        prev_is_default = df['last_login_date_previous'] == default_date
        both_default = curr_is_default & prev_is_default

        # Case 1: current is default
        df.loc[curr_is_default & ~both_default, 'days_until_month_end'] = 45
        df.loc[curr_is_default & ~both_default, 'last_login_interval'] = df.loc[curr_is_default & ~both_default].apply(
            lambda row: 45 + (calendar.monthrange(row['last_login_date_previous'].year, row['last_login_date_previous'].month)[1] - row['last_login_date_previous'].day)
            if row['last_login_date_previous'] != default_date else 90,
            axis=1
        )

        # Case 2: previous is default
        df.loc[prev_is_default & ~both_default, 'days_until_month_end'] = 45
        df.loc[prev_is_default & ~both_default, 'last_login_interval'] = df.loc[prev_is_default & ~both_default].apply(
            lambda row: 45 + calendar.monthrange(row['last_login_date_current'].year, row['last_login_date_current'].month)[1]
            if row['last_login_date_current'] != default_date else 90,
            axis=1
        )

        # Case 3: both are default
        df.loc[both_default, 'days_until_month_end'] = 90
        df.loc[both_default, 'last_login_interval'] = 90

        # Drop original date columns
        for col in ['last_login_date_previous', 'last_login_date_current', 'registration_init_time']:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

    # Label encode categorical features
    cat_cols = ['gender', 'city', 'registered_via']
    fitted_label_encoders = {}

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                fitted_label_encoders[col] = le
            else:
                le = label_encoders[col]
                df[col] = le.transform(df[col])

    # Separate target if training
    if is_train:
        y = df['is_churn'].astype(int)
        df.drop(columns=['is_churn'], inplace=True)
    else:
        y = None
        df.drop(columns=['is_churn'], inplace=True)

    # Standardize numerical features
    X = df.copy()
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    
    if is_train:
        return X_scaled, y, scaler, fitted_label_encoders
    else:
        return X_scaled
