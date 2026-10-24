import sys
import os
from sqlalchemy import Column, Integer, Float, DateTime, Boolean
from datetime import datetime

sys.path.append(os.path.abspath("."))

from config.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    transaction_amount = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    prediction_time_ms = Column(Float, nullable=False)
