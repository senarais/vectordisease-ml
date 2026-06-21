from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Inisialisasi Aplikasi
app = FastAPI(
    title="Vector Disease Early Warning API",
    description="API for predicting vector-borne disease risk based on climate data.",
    version="1.0.0"
)

# Load Model saat server pertama kali nyala (biar zero-latency pas di-hit)
try:
    model = joblib.load('ridge_model.pkl')
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {str(e)}")

# Skema Request Body (Validasi mirip TypeScript/Joi)
class WeatherPayload(BaseModel):
    temperature_celsius: float
    precipitation_mm: float
    extreme_weather_events: float
    heat_wave_days: float

@app.get("/")
def read_root():
    return {"message": "System is Running. Hit /api/predict-risk for predictions."}

@app.post("/api/predict-risk")
def predict_risk(data: WeatherPayload):
    try:
        # Konversi payload ke format array 2D yang dipahami scikit-learn
        features = np.array([[
            data.temperature_celsius,
            data.precipitation_mm,
            data.extreme_weather_events,
            data.heat_wave_days
        ]])
        
        # Eksekusi prediksi
        raw_prediction = model.predict(features)[0]
        
        # Format hasil agar rapi dan tidak ada angka negatif
        final_score = max(0.0, round(float(raw_prediction), 2))
        
        # Logika Status untuk memudahkan Frontend
        if final_score < 30:
            status = "Aman (Hijau)"
        elif final_score < 70:
            status = "Waspada (Kuning)"
        else:
            status = "Bahaya (Merah) - Butuh Mitigasi"

        # Return format JSON
        return {
            "success": True,
            "data": {
                "input_weather": data.dict(),
                "predicted_risk_score": final_score,
                "warning_status": status
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
