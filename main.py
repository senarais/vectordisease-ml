from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Vector Disease Early Warning API",
    version="1.0.0"
)

# Load Model
try:
    model = joblib.load('ridge_model.pkl')
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {str(e)}")

# Mount folder 'frontend' agar file static seperti CSS/JS internal (jika ada) bisa terbaca
# Pastikan folder frontend ada di direktori yang sama
if os.path.isdir("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

class WeatherPayload(BaseModel):
    temperature_celsius: float
    precipitation_mm: float
    extreme_weather_events: float
    heat_wave_days: float

# Route untuk menampilkan index.html UI
@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

# Endpoint API Backend
@app.post("/api/predict-risk")
def predict_risk(data: WeatherPayload):
    try:
        features = np.array([[
            data.temperature_celsius,
            data.precipitation_mm,
            data.extreme_weather_events,
            data.heat_wave_days
        ]])
        
        raw_prediction = model.predict(features)[0]
        final_score = max(0.0, round(float(raw_prediction), 2))
        
        if final_score < 30:
            status = "Aman (Hijau)"
        elif final_score < 70:
            status = "Waspada (Kuning)"
        else:
            status = "Bahaya (Merah) - Butuh Mitigasi"

        return {
            "success": True,
            "data": {
                "input_weather": data.model_dump(),
                "predicted_risk_score": final_score,
                "warning_status": status
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
