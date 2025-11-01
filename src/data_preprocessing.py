import os
import yaml
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

def read_params(config_path):
    """Read parameters from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def preprocess_data(config_path):
    """Preprocess data for housing dataset."""
    config = read_params(config_path)

    processed_path = os.path.join("data", "processed", "processed_data.csv")
    df = pd.read_csv(processed_path)

    target_col = config["split"]["target_col"]
    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- One-hot encode categorical columns ---
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if config["preprocessing"]["encode_categorical"]:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # --- Train-test split ---
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- Scaling numerical features ---
    if config["preprocessing"]["scale"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

    # --- Save all outputs ---
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    X_train.to_csv("data/train/X_train.csv", index=False)
    X_test.to_csv("data/test/X_test.csv", index=False)
    y_train.to_csv("data/train/y_train.csv", index=False)
    y_test.to_csv("data/test/y_test.csv", index=False)

    # --- Save scaler and encoded column structure ---
    dump(scaler, "models/scaler.joblib")
    dump(X.columns.tolist(), "models/feature_names.joblib")


    print("✅ Data Preprocessing Complete!")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Categorical columns encoded: {categorical_cols}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/params.yaml")
    args = parser.parse_args()
    preprocess_data(config_path=args.config)
