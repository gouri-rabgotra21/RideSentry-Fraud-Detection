from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# --- THIS IS THE CHANGE ---
# Create a FastAPI app instance with a formal title and description
app = FastAPI(
    title="RideSentry - Fraud Detection API",
    description="A real-time API to predict the probability of a ride-sharing transaction being fraudulent."
)
# --- END OF CHANGE ---

# Define the input data structure using Pydantic
class TripFeatures(BaseModel):
    fare: float
    seconds_since_signup: int
    user_trip_count: int
    user_trips_last_1h: int
    num_users_on_device: int
    device_degree: int
    card_degree: int

# Load the trained model
model = joblib.load('fraud_model.joblib')

# Create the prediction endpoint
@app.post("/predict")
def predict_fraud(trip: TripFeatures):
    # Create a pandas DataFrame from the input data
    feature_data = {
        'fare': [trip.fare],
        'seconds_since_signup': [trip.seconds_since_signup],
        'user_trip_count': [trip.user_trip_count],
        'user_trips_last_1h': [trip.user_trips_last_1h],
        'num_users_on_device': [trip.num_users_on_device],
        'device_degree': [trip.device_degree],
        'card_degree': [trip.card_degree]
    }
    prediction_features = pd.DataFrame(feature_data)

    # Make a prediction
    probability = model.predict_proba(prediction_features)[0][1]

    # Return the result
    return {"fraud_probability": probability}