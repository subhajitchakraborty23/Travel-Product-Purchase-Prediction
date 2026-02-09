from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Needed for frontend communication
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("model.pkl")

app = FastAPI(title="Travel Package Prediction API")

# --- FIX 1: ADD CORS MIDDLEWARE ---
# This allows your HTML file to communicate with this Python server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class TravelInput(BaseModel):
    Age: int
    TypeofContact: str
    CityTier: int
    DurationOfPitch: int
    Occupation: str
    Designation: str 
    Gender: str
    NumberOfPersonVisiting: int
    NumberOfFollowups: int
    ProductPitched: str
    PreferredPropertyStar: int
    MaritalStatus: str
    NumberOfTrips: int
    Passport: int
    PitchSatisfactionScore: int
    OwnCar: int
    NumberOfChildrenVisiting: int
    MonthlyIncome: int

@app.get("/")
def home():
    return {"message": "Travel Prediction API is running 🚀"}

# --- FIX 2: REMOVE DUPLICATE DECORATOR ---
@app.post("/predict")
def predict(data: TravelInput):
    # Convert Pydantic model to dictionary
    input_df = pd.DataFrame([data.dict()])

    # FEATURE ENGINEERING
    input_df["TotalVisits"] = (
        input_df["NumberOfPersonVisiting"]
        + input_df["NumberOfChildrenVisiting"]
    )

    # DROP ORIGINAL COLUMNS
    input_df.drop(
        ["NumberOfPersonVisiting", "NumberOfChildrenVisiting"],
        axis=1,
        inplace=True
    )

    # Ensure column order matches the training data exactly
    prediction = model.predict(input_df)[0]

    return {"ProdTaken": int(prediction)}