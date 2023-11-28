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

# Predict the next n days
n = 1211

sequence_length = 60

last_sequence = test_data[:sequence_length, :]
next_n_days_predictions = []

for _ in range(n):
    prediction = model.predict(np.reshape(last_sequence, (1, sequence_length, 1)))
    next_n_days_predictions.append(prediction[0, 0])
    last_sequence = np.concatenate((last_sequence[1:], prediction), axis=0)

# Scale the predictions back to original range
next_n_days_predictions = scaler.inverse_transform(np.array(next_n_days_predictions).reshape(-1, 1))

import datetime
# Generate dates starting from 01/08/2023
start_date = datetime.datetime(2020, 4, 4)
next_n_days_dates = [start_date + datetime.timedelta(days=i) for i in range(n)]

# Converting to pandas DataFrame
predictions_dates_df = pd.DataFrame(next_n_days_dates)
predictions_prices_df = pd.DataFrame(next_n_days_predictions)

df_predictions = pd.concat([predictions_dates_df, predictions_prices_df], axis=1)
df_predictions.columns = ['Date', "Price"]

forecast = df_predictions[df_predictions['Date'].isin(df.iloc[training_data_len: , :]["Date"])]

# Start the index from a particular number (e.g., 10)
start_index = 3337
new_index = range(start_index, start_index + len(forecast))

# Set the new index for the DataFrame
forecast.index = new_index

# Changing Column names
forecast.columns = ["Date", "Close"]

# Calculating RMSE
rmse=np.sqrt(np.mean(((forecast["Close"] - df.iloc[training_data_len: , :]["Close"])**2)))
rmse = round(rmse, 2)

# Calculating MAE
mae = np.mean(np.abs(forecast["Close"] - df.iloc[training_data_len: , :]["Close"]))
mae = round(mae, 2)

# Calculating MAPE
mape = np.mean(np.abs((forecast["Close"] - df.iloc[training_data_len: , :]["Close"]) / df.iloc[training_data_len: , :]["Close"])) * 100
mape = round(mape, 2)

st.subheader("Value of Root mean squared error, ")
st.write(f'RMSE = {rmse}')

st.subheader("Value of Mean Absolute Error, ")
st.write(f'MAE = {mae}')

st.subheader("Value of Mean Absolute Percentage error, ")
st.write(f'MAPE = {mape}%')

#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = forecast["Close"]

#Visualize the data
print(stock + "-")
fig1 = plt.figure(figsize=(10,4))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
st.pyplot(fig1)



# ///////////////////////////
from datetime import datetime
# Small Dataset
st.title("Small Dataset")

# Making Streamlit wep interface
# Title of web app
st.title("Stock Price Prediction Using LSTM")

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
st.subheader("Data for current year")
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

# Predict the next n days
n = 42

sequence_length = 20

test_data = scaled_data[training_data_len - 20: , : ]

last_sequence = test_data[:sequence_length, :]
next_n_days_predictions = []

for _ in range(n):
    prediction = model.predict(np.reshape(last_sequence, (1, sequence_length, 1)))
    next_n_days_predictions.append(prediction[0, 0])
    last_sequence = np.concatenate((last_sequence[1:], prediction), axis=0)

# Scale the predictions back to original range
next_n_days_predictions = scaler.inverse_transform(np.array(next_n_days_predictions).reshape(-1, 1))

import datetime
# Generate dates starting from 01/08/2023
start_date = datetime.datetime(2023, 6, 17)
next_n_days_dates = [start_date + datetime.timedelta(days=i) for i in range(n)]

# Converting to pandas DataFrame
predictions_dates_df = pd.DataFrame(next_n_days_dates)
predictions_prices_df = pd.DataFrame(next_n_days_predictions)

df_predictions = pd.concat([predictions_dates_df, predictions_prices_df], axis=1)
df_predictions.columns = ['Date', "Price"]

forecast = df_predictions[df_predictions['Date'].isin(df.iloc[training_data_len: , :]["Date"])]

# Start the index from a particular number (e.g., 10)
start_index = 115
new_index = range(start_index, start_index + len(forecast))

# Set the new index for the DataFrame
forecast.index = new_index

# Changing Column names
forecast.columns = ["Date", "Close"]

# Calculating RMSE
rmse=np.sqrt(np.mean(((forecast["Close"] - df.iloc[training_data_len: , :]["Close"])**2)))
rmse = round(rmse, 2)

# Calculating MAE
mae = np.mean(np.abs(forecast["Close"] - df.iloc[training_data_len: , :]["Close"]))
mae = round(mae, 2)

# Calculating MAPE
mape = np.mean(np.abs((forecast["Close"] - df.iloc[training_data_len: , :]["Close"]) / df.iloc[training_data_len: , :]["Close"])) * 100
mape = round(mape, 2)

st.subheader("Value of Root mean squared error, ")
st.write(f'RMSE = {rmse}')

st.subheader("Value of Mean Absolute Error, ")
st.write(f'MAE = {mae}')

st.subheader("Value of Mean Absolute Percentage error, ")
st.write(f'MAPE = {mape}%')

#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = forecast["Close"]

#Visualize the data
print(stock + "-")
fig1 = plt.figure(figsize=(10,4))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
st.pyplot(fig1)