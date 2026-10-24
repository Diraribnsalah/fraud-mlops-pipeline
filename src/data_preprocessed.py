import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import os

def load_and_clean_data(filepath = "data/creditcard.csv") :
  """
  load data and drop duplicate rows
  """
  df = pd.read_csv(filepath)
  initial_shape = df.shape[0]
  df = df.drop_duplicates()
  joblib.dump(df , "data/cleaned_data.csv")
  print(f"the initial shape is {initial_shape}\n the shape after droping is {df.shape[0]} ")
  return df
 

def preprocess_and_split(df , target_col = "Class" , test_size = 0.2 , use_smote = True) :
  X = df.drop(columns = [target_col]).astype(np.float32)
  y = df[target_col].values
  # split the data before scaling for avoiding data leakage problem
  X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = test_size , random_state = 42)
  scaler = RobustScaler()
   
  X_train[["Amount" , "Time"]] = scaler.fit_transform(X_train[["Amount" , "Time"]])
  X_test[["Amount" , "Time"]] = scaler.transform(X_test[["Amount" , "Time"]])
  test_data = X_test.copy()
  test_data[target_col] = y_test
  test_data.to_csv("data/test_data.csv", index=False)
  os.makedirs("models" , exist_ok = True)
  joblib.dump(scaler , "models/scaler.joblib")
  if use_smote :
    smote = SMOTE(sampling_strategy = 0.1 , random_state = 42)
    X_train_resampled , y_train_resampled = smote.fit_resample(X_train , y_train)
    print(f"the shape of train data after smote is {X_train_resampled.shape}\nthe sum of fraud is {np.sum(y_train_resampled == 1)}")
    return X_train_resampled , X_test , y_train_resampled , y_test , scaler
  return X_train , X_test , y_train , y_test , scaler  
 
