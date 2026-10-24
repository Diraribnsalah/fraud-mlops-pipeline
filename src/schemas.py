from pydantic import BaseModel, Field
from typing import List

class TransactionInput(BaseModel):
    features: List[float] = Field(..., min_items=30, max_items=30, description="List of 30 numerical features")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    status: str
    latency_ms: float
