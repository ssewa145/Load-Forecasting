import time
import datetime
import xgboost as xgb
import tensorflow as tf
import pandas as pd
import streamlit as st
import numpy
import sys
from bokeh.plotting import figure
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
  
  #df.drop(['T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day','date'], axis=1, inplace=True)
  #df
  #plt.plot(datetime, nat_demand, label = "nat_demand")
  #plt.plot(datetime, predictions, label = "predictions")
  #plt.xlabel('Date_Time')
  #plt.xlabel('Demand(MW)')
  #plt.show()


  x = 'datetime'
  y = 'predictions'

  p = figure(
      title='simple line example',
      x_axis_label='x',
      y_axis_label='y')

  p.line(x, y, legend_label='prediction', line_width=2)

  st.bokeh_chart(p, use_container_width=True)
  #if st.button("Click here to make the Peak Demand Prediction", key=3):
    #st.write(df)

run = web_app()
