import time
import datetime
import xgboost as xgb
import tensorflow as tf
import pandas as pd
import streamlit as st
import numpy
import sys
import matplotlib
from datetime import timedelta, date
from matplotlib import pyplot as plt
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



def web_app():
  st.write("""
  # Load Forecast Web App
  ## This app predicts the load to be supplied by the utility
  """)
  uploaded_file = st.file_uploader("Drag and drop the file here")
  if st.button("Submit"):
      df = pd.read_csv(uploaded_file)
      df1 = df.copy()
      df1.drop(['datetime','date','nat_demand'], axis=1, inplace=True)
      x_peak_demand = df1.copy()
      col_names = ['T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day']
      factors = x_peak_demand[col_names]
      scaler = StandardScaler().fit(factors.values)
      factors = scaler.transform(factors.values)
      x_peak_demand[col_names] = factors
      df['predictions'] = model_j.predict(x_peak_demand)

      #x_peak_demand[col_names] = factors
      #print(x_peak_demand)
      #df['prediction'] = model_j.predict(x_peak_demand)
      #df
      df2 = df.copy()
      df2.drop(['T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day','date'], axis=1, inplace=True)
      df2
      actual = df2['nat_demand']
      forecast = df2['predictions']
      datetime = df2['datetime']
      plt.plot(datetime, actual, label = "actual")
      plt.plot(datetime, forecast, label = "forecast")
      plt.xlabel('Date_Time')
      plt.xlabel('Demand(MW)')
      plt.show()
      #st.line_chart(df, 'datetime', ['natdemand','predictions'], width=10, height=10, use_container_width=True)
      df3 = df.copy()
      DOP_1 = df3.iat[0,20]
      DOP_1
      #while DOP_1 != 0:
      DOP = str(DOP_1)
      DOP = datetime.datetime.strptime(DOP, '%d/%m/%Y')
        #df = pd.read_csv(uploaded_file)
      df3["datetime"] = pd.to_datetime(df3["datetime"])
      df3.set_index("datetime").head(2)
      df4 = df[df3["datetime"].between(DOP,DOP + timedelta(hours=24))]
      result_peak = numpy.amax(df4['predictions'])
      st.text_area(label='Peak demand for the day is:- ',value=result_peak , height= 100,)
    #try :
        #DOP = str(DOP)
        #DOP = datetime.datetime.strptime(DOP, "%d/%m/%Y")
        #break
    #except ValueError:
        #print("Error: must be format dd/mm/yyyy ")
        #userkey = st.input("press 1 to try again or 0 to exit:")
        #if userkey == "0":
            #sys.exit()
  
 # df = pd.read_csv("continuous dataset.csv")
  #df["datetime"] = pd.to_datetime(df["datetime"])
  #df.set_index("datetime").head(2)
  #df1 = df[df["datetime"].between(DOP,DOP + timedelta(hours=24))]
  #df1.drop(['datetime','date','nat_demand'], axis=1, inplace=True)
  #x_peak_demand = df1.copy()
  #col_names = ['T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day']
  #factors = x_peak_demand[col_names]
  #scaler = StandardScaler().fit(factors.values)
  #factors = scaler.transform(factors.values)
  #x_peak_demand[col_names] = factors
  #df1['predictions'] = model_j.predict(x_peak_demand)
  #result_peak = numpy.amax(df1['predictions'])



run = web_app()
