import os
import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV file
        df = pd.read_csv(file.file)

        # Run prediction
        preds = model.predict(df)

        # Convert to list for JSON response
        results = preds.tolist()

        return {"predictions": results}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run app locally (Render overrides with $PORT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # use Render's $PORT or default 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
