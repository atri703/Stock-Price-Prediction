# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf # Importing finance stock datasets
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from pandas_datareader import data as pdr

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler


# Making Streamlit wep interface
# Title of web app
st.title("Stock Price Prediction Using LSTM")

# Setting date
start = "2007-01-01"
end = "2023-07-31"

# Convert start and end dates to datetime objects
start_date = datetime.strptime(start, "%Y-%m-%d")
end_date = datetime.strptime(end, "%Y-%m-%d")

# Set up the data reader with Yahoo Finance
yf.pdr_override()

# Selecting Preferred stocks from given list
stock_name_list = ("Apple", "Microsoft", "Amazon", "Netflix", "Infosys", "Adobe", "Google", "Nvidia")
stocks = ("AAPL", "MSFT", "AMZN", "NFLX", "INFY", "ADBE", "GOOGL", "NVDA")

stock_name = st.selectbox('Select dataset for prediction', stock_name_list)

index = stock_name_list.index(stock_name)

stock = stocks[index]

df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
df = df.reset_index()

# Looking at the data
st.subheader("Data from 2007 - 2023")
st.write(df.head())

# Describing the data
st.subheader("Describing the data")
st.write(df.describe())

# Data Visualization
st.subheader("Closing Price Graph")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.ylabel('Close')
plt.xlabel(None)
st.pyplot(fig)

#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])

#Converting the dataframe to a numpy array
data = data.values
data = data.reshape(-1, 1)

#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(data) *.8)


# Load the Keras model
model_filename = f'keras_model_{stock}.h5'
model_path = f"lstm_models/{model_filename}"
model = tf.keras.models.load_model(model_path)


# Scalling Data for better results and making it suiatble for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
# here we are Scaling the all of the data to be values between 0 and 1
scaled_data = scaler.fit_transform(data)

# //////
# here we are testing data set
test_data = scaled_data[training_data_len - 60: , : ]

#Creating the x_test and y_test data sets
x_test = []
y_test =  data[training_data_len : , : ]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

# here we are converting x_test to a numpy array
x_test = np.array(x_test)

# here we are reshaping the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# ///
# now we are getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#Undo scaling

# Calculating RMSE
rmse=np.sqrt(np.mean(((predictions - y_test)**2)))
rmse = round(rmse, 2)

# Calculating MAE
mae = np.mean(np.abs(predictions - y_test))
mae = round(mae, 2)

# Calculating MAPE
mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100
mape = round(mape, 2)

st.subheader("Value of Root mean squared error, ")
st.write(f'RMSE = {rmse}')

st.subheader("Value of Mean Absolute Error, ")
st.write(f'MAE = {mae}')

st.subheader("Value of Mean Absolute Percentage error, ")
st.write(f'MAPE = {mape}%')


# //////
#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the data
print(stock + "-")
fig2 = plt.figure(figsize=(10,4))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
st.pyplot(fig2)


# /////////////////////////////////////////////////////
# Case 2 Small dataset 

st.title("Small Dataset")

# Setting date
start = "2023-01-01"
end = "2023-07-31"

# Convert start and end dates to datetime objects
start_date = datetime.strptime(start, "%Y-%m-%d")
end_date = datetime.strptime(end, "%Y-%m-%d")

# Set up the data reader with Yahoo Finance
yf.pdr_override()

df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
df = df.reset_index()

# Looking at the data
st.subheader("Data for current")
st.write(df.head())

# Describing the data
st.subheader("Describing the data")
st.write(df.describe())

# Data Visualization
st.subheader("Closing Price Graph")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.ylabel('Close')
plt.xlabel(None)
st.pyplot(fig)

#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])

#Converting the dataframe to a numpy array
data = data.values
data = data.reshape(-1, 1)

#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(data) *.8)


# Load the Keras model
model_filename = f'keras_model_small_{stock}.h5'
model_path = rf'D:\Assingments\Dissertaion\models\both_cases_more_epochs\{model_filename}'
model = tf.keras.models.load_model(model_path)


# Scalling Data for better results and making it suiatble for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
# here we are Scaling the all of the data to be values between 0 and 1
scaled_data = scaler.fit_transform(data)

# //////
# here we are testing data set
test_data = scaled_data[training_data_len - 20: , : ]

#Creating the x_test and y_test data sets
x_test = []
y_test =  data[training_data_len : , : ]
for i in range(20,len(test_data)):
    x_test.append(test_data[i-20:i,0])

# here we are converting x_test to a numpy array
x_test = np.array(x_test)

# here we are reshaping the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# ///
# now we are getting the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#Undo scaling

# Calculating RMSE
rmse=np.sqrt(np.mean(((predictions - y_test)**2)))
rmse = round(rmse, 2)

# Calculating MAE
mae = np.mean(np.abs(predictions - y_test))
mae = round(mae, 2)

# Calculating MAPE
mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100
mape = round(mape, 2)

st.subheader("Value of Root mean squared error, ")
st.write(f'RMSE = {rmse}')

st.subheader("Value of Mean Absolute Error, ")
st.write(f'MAE = {mae}')

st.subheader("Value of Mean Absolute Percentage error, ")
st.write(f'MAPE = {mape}%')


# //////
#Creating a new dataframe with only the 'Close' column



data = df.filter(['Close'])

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the data
print(stock + "-")
fig2 = plt.figure(figsize=(10,4))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
st.pyplot(fig2)