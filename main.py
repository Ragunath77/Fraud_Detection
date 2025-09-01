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

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "Fraud detection API is live 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read CSV
        df = pd.read_csv(file.file)

        # Make sure columns match model
        required_columns = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns
        df = df[required_columns]

        # Predict
        preds = model.predict(df)
        df["prediction"] = preds

        # Convert CSV to memory
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        # Return CSV for download
        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
