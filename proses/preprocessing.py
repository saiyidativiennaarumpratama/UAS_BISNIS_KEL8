import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

X_pre = pd.read_csv("data/dataset_garam(droplabel).csv")
# y = X_pre['Grade']

def minMax():
  st.write('Data Awal Sebelum di lakukan Preprocessing')
  st.dataframe(X_pre)
  
  st.write('Data setelah dilakukan Preprocessing menggunakan Min-Max Scaler')
  scaler = MinMaxScaler()
  data_scaled = scaler.fit_transform(X_pre)

  #memasukan fitur 
  features_names = X_pre.columns.copy()
  scaled_features = pd.DataFrame(data_scaled, columns=features_names)
  st.write(scaled_features)

  # Save Scaled
  scaler_filename = "df_scaled(norm).save"
  joblib.dump(scaler, scaler_filename)
 
