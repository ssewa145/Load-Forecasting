# -*- coding: utf-8 -*-
"""streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Osc_xxHZkLeTjc5KT3qYLN2gmTDIkMG6
"""

import datetime
import xgboost as xgb
import tensorflow as tf
import pandas as pd
import streamlit as st
import numpy
import sys
from datetime import timedelta, date
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
from tensorflow.keras.optimizers import Adam as adam
from tensorflow.python.keras.losses import MeanSquaredLogarithmicError

from keras.initializers import glorot_uniform
#Reading the model from JSON file
with open('loadfc_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
#load the model architecture 
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.summary()
model_j.load_weights('loadfc_weights.h5')

#compiling the model
msle = MeanSquaredLogarithmicError()
model_j.compile(
    loss=msle, 
    # optimizer=adam(learning_rate=learning_rate), 
    optimizer='adam', 
    metrics=[msle]
)

def peak_app():
  st.write("""
  # Peak Load
  ## This will determine the peak demand for a given day
  """)
  st.header("The date input is required")
  ##while True :
  DOP = st.date_input('Date of Prediction ', datetime.datetime.strptime(2019, 7, 6, "%d/%m/%Y"))
                       
    #try :
        #DOP = str(DOP)
        #DOP = datetime.datetime.strptime(DOP, "%d/%m/%Y")
        #break
    #except ValueError:
        #print("Error: must be format dd/mm/yyyy ")
        #userkey = st.input("press 1 to try again or 0 to exit:")
        #if userkey == "0":
            #sys.exit()
  
  df = pd.read_csv("continuous dataset.csv")
  df["datetime"] = pd.to_datetime(df["datetime"])
  df.set_index("datetime").head(2)
  df[df["datetime"].between(DOP,DOP + timedelta(hours=24))]
  df.drop(['datetime','date','nat_demand'], axis=1, inplace=True)
  x_peak_demand = df.copy()
  col_names = ['T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day']
  factors = x_peak_demand[col_names]
  scaler = StandardScaler().fit(factors.values)
  factors = scaler.transform(factors.values)
  x_peak_demand[col_names] = factors
  result_peak = numpy.amax(model_j.predict(x_peak_demand))


  if st.button("Click here to make the Peak Demand Prediction"):
    st.text_area(label='Load prediction is:- ',value=result_peak , height= 100)


def web_app():
  st.write("""
  # Load Forecast Web App
  ## This app predicts the load to be supplied by the utility
  """)
  st.header("User Details")
  st.subheader("Kindely Enter The following Details in order to make a prediction")
  T2M_toc1 = st.number_input("Temperature of Tocumen")
  T2M_toc1 = (T2M_toc1-27.39811537002067)/1.6761489507888916
  QV2M_toc1 = st.number_input("Relative Humidity of Tocumen")
  QV2M_toc1 = (QV2M_toc1-0.018313160032571406)/0.0016069496418865945
  TQL_toc1 = st.number_input("Liquid Precipitation of Tocumen")
  TQL_toc1 = (TQL_toc1-0.08003988163517133)/0.06548029809978714
  W2M_toc1 = st.number_input("Wind Speed of Tocumen")
  W2M_toc1 = (W2M_toc1-13.400886189856779)/7.328792527364909
  T2M_san1 = st.number_input("Temperature of San")
  T2M_san1 = (T2M_san1-26.91410107611334)/3.0226171870424743
  QV2M_san1 = st.number_input("Relative Humidity of San")
  QV2M_san1 = (QV2M_san1-0.017842629823295996)/0.0018943023405940354
  TQL_san1 = st.number_input("Liquid Precipitation of San")
  TQL_san1 = (TQL_san1-0.10629538179184062)/0.08632141691484443
  W2M_san1 = st.number_input("Wind Speed of San")
  W2M_san1 = (W2M_san1-7.068363287679824)/4.133478481989156
  T2M_dav1 = st.number_input("Temperature of Dav")
  T2M_dav1 = (T2M_dav1-24.715383019754064)/2.4170100615515855
  QV2M_dav1 = st.number_input("Relative Humidity of Dav")
  QV2M_dav1 = (QV2M_dav1-0.016863202009837555)/0.0015874588496469595
  TQL_dav1 = st.number_input("Liquid Precipitation of Dav")
  TQL_dav1 = (TQL_dav1-0.14450035478452447)/0.08760944618689656
  W2M_dav1 = st.number_input("Wind Speed of Dav")
  W2M_dav1 = (W2M_dav1-3.5830389455709915)/1.7146204489974868
  Holiday_ID1 = st.number_input("Nature of Holiday")
  Holiday_ID1 = (Holiday_ID1-0.6926797007398834)/3.1245110303631227
  holiday1 = st.number_input("Holiday")
  holiday1 = (holiday1-0.0618774025544579)/0.24093772868283403
  school1 = st.number_input("School Holiday")
  school1 = (school1-0.7252097714214856)/0.4464177374536607
  hour1 = st.number_input("Hour of Day")
  hour1 = (hour1-11.462323812673088)/6.905290140015345
  month1 = st.number_input("Month")
  month1 = (month1-6.242094820815939)/3.4426341567092456
  day1 = st.number_input("Day of the Week")
  day1 = (day1-3.9887984127640226)/2.000950133623064
  
 
  #import pandas as pd

  data=[[T2M_toc1,QV2M_toc1,TQL_toc1,W2M_toc1,T2M_san1,QV2M_san1,TQL_san1,W2M_san1,T2M_dav1,QV2M_dav1,TQL_dav1,W2M_dav1,Holiday_ID1,holiday1,school1,hour1,month1,day1]]

  df1=pd.DataFrame(data,columns=['T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day'])

  df1
    
  #data={'T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day':[T2M_toc,QV2M_toc,TQL_toc,W2M_toc,T2M_san,QV2M_san,TQL_san,W2M_san,T2M_dav,QV2M_dav,TQL_dav,W2M_dav,Holiday_ID,holiday,school,hour,month,day]}

  #df=pd.dataframe(data)
                  
  #,index=[‘No.1’,’No.2’,’No.3’,’No.4’,'No.5','No.6','No.7','No.8','No.9','No.10','No.11','No.12','No.13','No.14','No.15','No.16','No.17,'No.18'])

#dfe
  
  
  result = model_j.predict(df1)
  
  if st.button("Click here to make the Prediction"):
    st.text_area(label='Load prediction is:- ',value=result , height= 100)
  
  
if st.button("Click here to determine PEAK demand for the day", key=1):
  run = peak_app()
if st.button("Click here to determine HOURLY demand for the day", key=2):
  run = web_app()
#if st.button("Click here to determine PEAK demand for the day", key=1):
 #   run = peak_app()
#if st.button("Click here to determine HOURLY demand for the day", key=2):
 #   run = web_app()

