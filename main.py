from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
import io

app = FastAPI()

# Load model at startup
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded CSV into pandas
    contents = await file.read()
    data = pd.read_csv(io.BytesIO(contents))

    # Ensure columns match model's expectations
    model_features = model.get_booster().feature_names
    missing_cols = set(model_features) - set(data.columns)
    for c in missing_cols:
        data[c] = 0
    extra_cols = set(data.columns) - set(model_features)
    data = data.drop(columns=extra_cols)
    data = data[model_features]

    # Run predictions
    preds = model.predict(data)
    data["Prediction"] = preds

    # Return JSON
    return data.to_dict(orient="records")
