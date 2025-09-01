import os
import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO

# Load model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create app
app = FastAPI()

# Enable CORS (important for frontend → backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (for dev; replace "*" with frontend URL in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route (for Render/browser testing)
@app.get("/")
def home():
    return {"status": "ok", "message": "Fraud detection API is live 🚀"}

# Prediction route returning CSV
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV file
        df = pd.read_csv(file.file)

        # Run prediction
        preds = model.predict(df)

        # Add predictions to dataframe
        df["prediction"] = preds

        # Convert dataframe to CSV in-memory
        stream = StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        # Return as downloadable CSV
        response = StreamingResponse(stream, media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
