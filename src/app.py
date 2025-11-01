from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.inference_preprocessing import preprocess_for_inference

# Initialize FastAPI app
app = FastAPI(title="Housing Price Prediction API", version="1.0")

# Load your trained model
model = joblib.load("models/LinearRegression_model.pkl")

# Define the input structure (same features you used during training)
class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

# Root endpoint (health check)
@app.get("/")
def home():
    return {"message": "Welcome to the Housing Price Prediction API!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: HouseData):
    # Convert input into DataFrame
    df = pd.DataFrame([data.dict()])

    # Preprocessing (convert categorical columns same way as training)
    for col in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]:
        df[col] = df[col].map({"yes": 1, "no": 0, "furnished": 2, "semi-furnished": 1, "unfurnished": 0})

    # Predict
    df_processed = preprocess_for_inference(df)
    prediction = model.predict(df_processed)
    return {"predicted_price": round(float(prediction[0]), 2)}
