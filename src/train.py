import sys
import os

#  Add src directory to system path for modular imports
sys.path.append(os.path.abspath("src"))

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import onnx
import onnxmltools
from skl2onnx import convert_sklearn

# Import separate FloatTensorTypes to prevent type mismatch runtime errors
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMLFloatTensorType

from data_preprocessed import load_and_clean_data, preprocess_and_split

# 2. Set DagsHub and MLflow tracking credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "PUT_YOUR_MLFLOW_USERNAME_HERE"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "PUT_YOUR_MLFLOW_TOKEN_HERE") 

mlflow.set_tracking_uri("https://dagshub.com/diraribnsalah/my-first-repo.mlflow")
mlflow.set_experiment("Credit_Card_Fraud_Detection_Preprocessed")

def save_as_onnx(model, model_type, num_features, output_path="models/fraud_model.onnx"):
    """
    Serializes and exports the best trained model to ONNX format.
    """
    os.makedirs("models", exist_ok=True)
    
    if model_type == "rf":
        initial_type = [("float_input", SklearnFloatTensorType([None, num_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
    elif model_type == "xgb":
        initial_type = [("float_input", OnnxMLFloatTensorType([None, num_features]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        
    elif model_type == "lgbm":
        initial_type = [("float_input", OnnxMLFloatTensorType([None, num_features]))]
        onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)
        
    onnx.save_model(onnx_model, output_path)
    print(f"\n Successfully exported best model ({model_type.upper()}) to ONNX at: {output_path}")

def train_and_evaluate():
    """
    Main pipeline: Loads preprocessed data, trains candidate models,
    logs experiments to MLflow/DagsHub, and exports the winning model.
    """
    print(" Loading and cleaning data...")
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df, use_smote=True)
    
    # Convert feature matrices to raw contiguous NumPy float32 arrays BEFORE fitting.
    # This prevents estimators from binding column names (V1..V28, Amount, Time),
    # ensuring seamless ONNX conversion and suppressing feature name warnings.
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    
    models = {
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1), 
            "rf"
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=10, random_state=42, n_jobs=-1), 
            "xgb"
        ),
        "LightGBM": (
            LGBMClassifier(random_state=42, is_unbalance=True, n_jobs=-1, verbose=-1), 
            "lgbm"
        )
    }
    
    best_pr_auc = 0.0
    best_model_info = None

    for model_name, (model, model_type) in models.items():
        with mlflow.start_run(run_name=f"{model_name}_Preprocessed"):
            print(f"\n Training model {model_name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Compute evaluation metrics
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Log hyperparameters and metrics to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("data_preprocessed", True)
            mlflow.log_param("smote_applied", True)
            
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            
            print(f" {model_name} Metrics:")
            print(f"   - PR-AUC (AUPRC) : {pr_auc:.4f}")
            print(f"   - Recall         : {recall:.4f}")
            print(f"   - Precision      : {precision:.4f}")
            print(f"   - F1-Score       : {f1:.4f}")
            
            # Select the best model based on PR-AUC
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_model_info = (model, model_type)

    if best_model_info:
        save_as_onnx(best_model_info[0], best_model_info[1], X_train.shape[1])

if __name__ == "__main__":
    train_and_evaluate()
