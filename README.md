# Vector Disease Early Warning System (VBD EWS) API

A fast, lightweight, and zero-latency machine learning API built with **FastAPI** to predict the risk of vector-borne diseases based on meteorological data. The underlying predictive engine utilizes a highly interpretable **Ridge Regression** model (`scikit-learn`), strictly regularized to handle climate multicollinearity.

## Features
- **High Performance:** Built on FastAPI & Uvicorn for asynchronous, blazingly fast request handling.
- **Interpretable AI:** Uses an L2-regularized linear model, prioritizing algorithmic transparency for medical informatics.
- **Docker-Ready:** Fully containerized for seamless deployment on any VPS or home server environment.
- **Frontend-Friendly:** Returns standardized, deeply nested JSON structures with ready-to-use risk status labels.

## Prerequisites
Ensure your server or local machine has the following installed:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Local / Home Server Deployment

1. **Clone or copy the repository** to your server:
   ```bash
   cd vbd-ews-api
Ensure the machine learning model is present:
Make sure the ridge_model.pkl file (exported from your training environment) is placed in the root of this directory.

Build and run the Docker container:

Bash
docker-compose up -d --build
Verify the deployment:
The API will now be running on http://localhost:8001. You can check the container status using:

Bash
docker ps
Cloudflare Tunnel Integration
To expose this API securely without opening router ports:

Go to your Cloudflare Zero Trust Dashboard > Access > Tunnels.

Add a new Public Hostname (e.g., api-cuaca.yourdomain.com).

Set the Service Type to HTTP and the URL to localhost:8001.

Save the tunnel. Your API is now securely live via HTTPS.

API Documentation
1. Health Check
Endpoint: GET /

Description: Checks if the server is running.

Response:

JSON
{
  "message": "System is Running. Hit /api/predict-risk for predictions."
}
2. Predict Risk Score
Endpoint: POST /api/predict-risk

Headers: Content-Type: application/json

Request Body (JSON):

JSON
{
  "temperature_celsius": 32.5,
  "precipitation_mm": 120.0,
  "extreme_weather_events": 2.0,
  "heat_wave_days": 1.0
}
Success Response (200 OK):

JSON
{
  "success": true,
  "data": {
    "input_weather": {
      "temperature_celsius": 32.5,
      "precipitation_mm": 120.0,
      "extreme_weather_events": 2.0,
      "heat_wave_days": 1.0
    },
    "predicted_risk_score": 75.4,
    "warning_status": "Bahaya (Merah) - Butuh Mitigasi"
  }
}
License and Authorship
Developed by Abisena Rais as part of a climate-based health informatics research initiative.
