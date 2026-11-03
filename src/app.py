import os
import sys
import time
from contextlib import asynccontextmanager
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from config.database import engine, Base, get_db, SessionLocal
from src.db_models import PredictionLog
from src.schemas import TransactionInput, PredictionResponse
 
Base.metadata.create_all(bind=engine)

ONNX_MODEL_PATH = "models/fraud_model.onnx"
SCALER_PATH = "models/scaler.joblib"

 
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(ONNX_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model artifacts not found in 'models/' directory.")
    
     
    app.state.ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    app.state.scaler = joblib.load(SCALER_PATH)
    app.state.input_name = app.state.ort_session.get_inputs()[0].name
    
    yield
    
   
    del app.state.ort_session
    del app.state.scaler

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="High-performance ONNX Inference Service with Background Logging",
    version="1.0.0",
    lifespan=lifespan
)

 
def log_prediction_to_db(amount: float, fraud_prob: float, is_fraud: bool, latency: float):
    db: Session = SessionLocal()
    try:
        log_entry = PredictionLog(
            transaction_amount=amount,
            fraud_probability=fraud_prob,
            is_fraud=is_fraud,
            prediction_time_ms=latency
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Failed to log prediction to DB: {e}")
    finally:
        db.close()

@app.get("/health")
def health_check():
    return {"status": "healthy", "onnx_model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(
    transaction: TransactionInput, 
    background_tasks: BackgroundTasks
):
    start_time = time.time()
    
     
    if len(transaction.features) < 30:
        raise HTTPException(status_code=422, detail="Transaction features must contain at least 30 elements.")

    try:
        raw_features = np.array(transaction.features, dtype=np.float32).reshape(1, -1)
        
         
        time_amount = raw_features[:, [0, 29]]
        scaled_time_amount = app.state.scaler.transform(time_amount).astype(np.float32)
        
        scaled_features = raw_features.copy()
        scaled_features[:, 0] = scaled_time_amount[:, 0]
        scaled_features[:, 29] = scaled_time_amount[:, 1]
        
         
        outputs = app.state.ort_session.run(None, {app.state.input_name: scaled_features})
        
        # Extracting the Probabilities of the fraud (Probabilities array)
        prob_fraud = float(outputs[1][0][1])
        is_fraud = prob_fraud >= 0.5
        latency_ms = round((time.time() - start_time) * 1000, 3)
  
        background_tasks.add_task(
            log_prediction_to_db,
            amount=float(transaction.features[29]),
            fraud_prob=prob_fraud,
            is_fraud=is_fraud,
            latency=latency_ms
        )

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(prob_fraud, 4),
            status="APPROVED" if not is_fraud else "FLAGGED_FRAUD",
            latency_ms=latency_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
