import pandas as pd
from joblib import load

def preprocess_for_inference(df: pd.DataFrame):
    """Apply same preprocessing used during training."""
    # Load trained artifacts
    scaler = load("models/scaler.joblib")
    feature_names = load("models/feature_names.joblib")

    # Same one-hot encoding logic
    df = pd.get_dummies(df, drop_first=True)

    # Add missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training
    df = df[feature_names]

    # Scale using the fitted scaler
    df_scaled = scaler.transform(df)

    return pd.DataFrame(df_scaled, columns=feature_names)
