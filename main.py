import io
import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# Load your trained model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def home():
    return {"status": "ok", "message": "Fraud detection API is live 🚀"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV
        df = pd.read_csv(file.file)

        # Run prediction
        preds = model.predict(df)
        df["prediction"] = preds  # append predictions as last column

        # Convert CSV to in-memory buffer
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        # Return CSV as downloadable file
        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
