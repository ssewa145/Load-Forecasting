import time
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
  """)
  st.header("The date input is required")
  ##while True :
  #DOP_1 = st.date_input("Enter the date in the format dd/mm/yyyy", datetime.datetime(%d/%m/%Y))
  #DOP = str(DOP_1)
  #DOP = datetime.datetime.strptime(DOP, '%d/%m/%Y')
  uploaded_file = st.file_uploader("Drag and drop the file here")
  DOP_1 = st.text_input("Enter the date in the format dd/mm/yyyy")
  #DOP_1 = uploaded_file.at[0, 'datetime']
  st.write('Please wait. The Software will determine the peak load of:', DOP_1)
  DOP = str(DOP_1)
  DOP = datetime.datetime.strptime(DOP, '%d/%m/%Y')
                       
    #try :
        #DOP = str(DOP)
        #DOP = datetime.datetime.strptime(DOP, "%d/%m/%Y")
        #break
    #except ValueError:
        #print("Error: must be format dd/mm/yyyy ")
        #userkey = st.input("press 1 to try again or 0 to exit:")
        #if userkey == "0":
            #sys.exit()
  
  df = pd.read_csv(uploaded_file)
  df.drop('hours', axis=1, inplace=True)
  df["datetime"] = pd.to_datetime(df["datetime"])
  df.set_index("datetime").head(2)
  df1 = df[df["datetime"].between(DOP,DOP + timedelta(hours=24))]
  df1.drop(['datetime','date','nat_demand'], axis=1, inplace=True)
  x_peak_demand = df1.copy()
  col_names = ['T2M_toc','QV2M_toc','TQL_toc','W2M_toc','T2M_san','QV2M_san','TQL_san','W2M_san','T2M_dav','QV2M_dav','TQL_dav','W2M_dav','Holiday_ID','holiday','school','hour','month','day']
  factors = x_peak_demand[col_names]
  scaler = StandardScaler().fit(factors.values)
  factors = scaler.transform(factors.values)
  x_peak_demand[col_names] = factors
  df1['predictions'] = model_j.predict(x_peak_demand)
  result_peak = numpy.amax(df1['predictions'])
  


  if st.button("Click here to make the Peak Demand Prediction", key=3):
    st.text_area(label='Peak demand for the day is:- ',value=result_peak , height= 100,)

run = peak_app()
