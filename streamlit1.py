# -*- coding: utf-8 -*-
"""streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Osc_xxHZkLeTjc5KT3qYLN2gmTDIkMG6
"""

import tensorflow as tf
import xgboost as xgb
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
  
  object = StandardScaler()
  object.fit_transform(T2M_toc)
  
  st.text_area(label='Load predition is:- ',value=input_data , height= 100)
  
 
   
run = web_app()
