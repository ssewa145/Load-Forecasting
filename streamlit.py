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
  T2M_toc = st.number_input("Temperature of Tocumen")
  T2M_toc = (T2M_toc-27.39811537002067)/1.6761489507888916
  QV2M_toc = st.number_input("Relative Humidity of Tocumen")
  QV2M_toc = (QV2M_toc-0.018313160032571406)/0.0016069496418865945
  TQL_toc = st.number_input("Liquid Precipitation of Tocumen")
  TQL_toc = (TQL_toc-0.08003988163517133)/0.06548029809978714
  W2M_toc = st.number_input("Wind Speed of Tocumen")
  W2M_toc = (W2M_toc-13.400886189856779)/7.328792527364909
  T2M_san = st.number_input("Temperature of San")
  T2M_san = (T2M_san-26.91410107611334)/3.0226171870424743
  QV2M_san = st.number_input("Relative Humidity of San")
  QV2M_san = (QV2M_san-0.017842629823295996)/0.0018943023405940354
  TQL_san = st.number_input("Liquid Precipitation of San")
  TQL_san = (TQL_san-0.10629538179184062)/0.08632141691484443
  W2M_san = st.number_input("Wind Speed of San")
  W2M_san = (W2M_san-7.068363287679824)/4.133478481989156
  T2M_dav = st.number_input("Temperature of Dav")
  T2M_dav = (T2M_dav-24.715383019754064)/2.4170100615515855
  QV2M_dav = st.number_input("Relative Humidity of Dav")
  QV2M_dav = (QV2M_dav-0.016863202009837555)/0.0015874588496469595
  TQL_dav = st.number_input("Liquid Precipitation of Dav")
  TQL_dav = (TQL_dav-0.14450035478452447)/0.08760944618689656
  W2M_dav = st.number_input("Wind Speed of Dav")
  W2M_dav = (W2M_dav-3.5830389455709915)/1.7146204489974868
  Holiday_ID = st.number_input("Nature of Holiday")
  Holiday_ID = (Holiday_ID-0.6926797007398834)/3.1245110303631227
  holiday = st.number_input("Holiday")
  holiday = (holiday-0.0618774025544579)/0.24093772868283403
  school = st.number_input("School Holiday")
  school = (school-0.7252097714214856)/0.4464177374536607
  hour = st.number_input("Hour of Day")
  hour = (hour-11.462323812673088)/6.905290140015345
  month = st.number_input("Month")
  month = (month-6.242094820815939)/3.4426341567092456
  day = st.number_input("Day of the Week")
  day = (day-3.9887984127640226)/2.000950133623064
  
  
  result = model.predict([['T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day']])
  
run = web_app()

if st.button("Press here to make Prediction"):
    st.text_area(label='Load predition is:- ',value=result , height= 100)
  
