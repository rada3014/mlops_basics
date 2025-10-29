import os
import yaml
import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def read_params(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def train_and_log_model(config_path):
    config = read_params(config_path)

    X_train = pd.read_csv("data/train/X_train.csv")
    y_train = pd.read_csv("data/train/y_train.csv").values.ravel()
    X_test = pd.read_csv("data/test/X_test.csv")
    y_test = pd.read_csv("data/test/y_test.csv").values.ravel()

    model_type = config["model"]["type"]
    model_dir = config["model"]["save_model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # choose model
    if model_type == "LinearRegression":
        model = LinearRegression()
    elif model_type == "RandomForest":
        model = RandomForestRegressor(random_state=config["model"]["random_state"])
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    # --- MLflow tracking starts ---
    mlflow.set_experiment("HousingPricePrediction")

    with mlflow.start_run(run_name=f"{model_type}_run"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # log parameters, metrics, and model
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # save model artifact
        model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"✅ MLflow Run logged for {model_type}")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/params.yaml")
    args = parser.parse_args()
    train_and_log_model(args.config)
