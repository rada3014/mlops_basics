import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    """Read parameters from YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_and_save_data(config_path):
    """Load data from raw path and save processed copy."""
    config = read_params(config_path)
    raw_path = config["data_path"]
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Example: reading a CSV (replace with your dataset)
    df = pd.read_csv(raw_path)

    # Minimal cleaning demo
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    processed_path = os.path.join(processed_dir, "processed_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"✅ Data saved to: {processed_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/params.yaml")
    args = parser.parse_args()
    load_and_save_data(config_path=args.config)
