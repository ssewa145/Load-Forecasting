# -*- coding: utf-8 -*-
"""streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Osc_xxHZkLeTjc5KT3qYLN2gmTDIkMG6
"""
import xgboost as xgb
import tensorflow as tf
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

from keras.initializers import glorot_uniform
#Reading the model from JSON file
with open('loadfc_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
#load the model architecture 
model = tf.keras.models.model_from_json(json_savedModel)


def web_app():
  st.write("""
  # Load Forecast Web App
  ## This app predicts the load to be supplied by the utility
  """)
  st.header("User Details")
  st.subheader("Kindely Enter The following Details in order to make a prediction")
  T2M_toc = st.number_input("Temperature of Tocumen",0,10)
  QV2M_toc = st.number_input("Relative Humidity of Tocumen",0,10)
  TQL_toc = st.number_input("Liquid Precipitation of Tocumen",0,10)
  W2M_toc = st.number_input("Wind Speed of Tocumen",0,10)
  T2M_san = st.number_input("Temperature of San",0,10)
  QV2M_san = st.number_input("Relative Humidity of San",0,10)
  TQL_san = st.number_input("Liquid Precipitation of San",0,10)
  W2M_san = st.number_input("Wind Speed of San",0,10)
  T2M_dav = st.number_input("Temperature of Dav",0,10)
  QV2M_dav = st.number_input("Relative Humidity of Dav",0,10)
  TQL_dav = st.number_input("Liquid Precipitation of Dav")
  W2M_dav = st.number_input("Wind Speed of Dav")
  Holiday_ID = st.number_input("Nature of Holiday")
  holiday = st.number_input("Holiday")
  school = st.number_input("School Holiday")
  hour = st.number_input("Hour of Day")
  month = st.number_input("Month")
  day = st.number_input("Day of the Week")
  
  input_data = [T2M_toc,QV2M_toc,TQL_toc,W2M_toc,T2M_san,QV2M_san,TQL_san,W2M_san,T2M_dav,QV2M_dav,TQL_dav,W2M_dav,Holiday_ID,holiday,school,hour,month,day]
  scaler = StandardScaler().fit(input_data.values)
  input_data = scaler.transform(input_data.values)

 
  
  result = model.predict([[input_data]])
  st.text_area(label='Load predition is:- ',value=result , height= 100)

if st.button("Press here to make Prediction"):
  run = web_app()
