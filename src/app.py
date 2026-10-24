import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import time
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from config.database import engine, Base, get_db
from src.db_models import PredictionLog
from src.schemas import TransactionInput, PredictionResponse

 
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="High-performance ONNX Inference Service with Prediction Persistence",
    version="1.0.0"
)

ONNX_MODEL_PATH = "models/fraud_model.onnx"
SCALER_PATH = "models/scaler.joblib"

if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model artifacts not found in 'models/' directory.")

ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
input_name = ort_session.get_inputs()[0].name

@app.get("/health")
def health_check():
    return {"status": "healthy", "onnx_model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionInput, db: Session = Depends(get_db)):
    start_time = time.time()
    
    try:
        raw_features = np.array(transaction.features, dtype=np.float32).reshape(1, -1)
        time_amount = raw_features[:, [0, 29]]
        
         
        scaled_time_amount = scaler.transform(time_amount).astype(np.float32)
        
         
        scaled_features = raw_features.copy()
        scaled_features[:, 0] = scaled_time_amount[:, 0]   # Time
        scaled_features[:, 29] = scaled_time_amount[:, 1]  # Amount
        
        outputs = ort_session.run(None, {input_name: scaled_features})
        
        prob_fraud = float(outputs[1][0][1]) if isinstance(outputs[1], list) else float(outputs[1][0][1])

        is_fraud = bool(prob_fraud >= 0.5)
        latency_ms = round((time.time() - start_time) * 1000, 3)

        log_entry = PredictionLog(
            transaction_amount=float(transaction.features[-1]),
            fraud_probability=prob_fraud,
            is_fraud=is_fraud,
            prediction_time_ms=latency_ms
        )
        db.add(log_entry)
        db.commit()

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(prob_fraud, 4),
            status="APPROVED" if not is_fraud else "FLAGGED_FRAUD",
            latency_ms=latency_ms
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
