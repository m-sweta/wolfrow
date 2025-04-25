from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import datetime
import os
import random
from sklearn.linear_model import LinearRegression
import numpy as np

# Constants and file paths
CSV_PATH = "workouts.csv"
WEEK_TYPES = ["base", "threshold", "taper"]

# FastAPI app
app = FastAPI()

# Sample workout data class
class Workout(BaseModel):
    date: datetime.datetime  # Date as datetime object (ISO format string parsed by FastAPI)
    distance: float
    time: float
    week_type: str

# Read workout data
def load_workout_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    else:
        return pd.DataFrame(columns=["date", "distance", "time", "week_type"])

# Endpoint to get workout history
@app.get("/history")
def get_history():
    df = load_workout_data()
    return df.to_dict(orient="records")

# Endpoint to log a new workout
@app.post("/log_workout")
def log_workout(workout: Workout):
    print(f"Received workout: {workout}")  # Add logging here
    df = load_workout_data()
    new_row = {
        "date": pd.to_datetime(workout.date),
        "distance": workout.distance,
        "time": workout.time,
        "week_type": workout.week_type
    }
    df = df.append(new_row, ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    return {"message": "Workout logged successfully!"}

# Endpoint for fatigue prediction (using pace)
@app.get("/fatigue_prediction")
def fatigue_prediction():
    df = load_workout_data()
    if len(df) < 2:
        return {"message": "Not enough data to predict fatigue."}
    
    df = df.sort_values("date")
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["time"].values  # Assuming time as a simple predictor
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    
    # Fatigue logic: if the latest pace is greater than the predicted trend pace by more than 5%, flag as fatigued
    fatigue = bool(y[-1] > trend[-1] * 1.05)  # Convert to native Python bool
    
    return {"fatigue": fatigue, "latest_pace": y[-1], "trend_pace": trend[-1]}

# Start server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

