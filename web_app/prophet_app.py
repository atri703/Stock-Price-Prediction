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

# Importing Prophet model
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Making Streamlit wep interface
# Title of web app
st.title("Stock Price Prediction Using Prophet")

# ////////////////////////////////////////////////////////////////////////////////
# Case 1: Large Dataset
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

# Describing the data
st.subheader("Data from 2007 - 2023")
st.write(df.head())

training_len = math.ceil(0.8 * df.shape[0])

stock_training_data = df.iloc[0:training_len, : ]

# Data Visualization
st.subheader("Closing Price Graph")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.ylabel('Close')
plt.xlabel(None)
st.pyplot(fig)

# Changing name of col to fit in prophet model
# Select only the important features i.e. the date and price
new_df = stock_training_data[["Date","Close"]] # select Date and Price
# Rename the features: These names are NEEDED for the model fitting
new_df = new_df.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset

# Fitting data into the model
m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(new_df) # fit the model using all data

# Make future predictions
future = m.make_future_dataframe(periods=1214) #we need to specify the number of days in future
forecast = m.predict(future)

forecast = forecast[forecast['ds'].isin(df.iloc[training_len:, :]["Date"])]

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.head())

rmse = np.sqrt(mean_squared_error(forecast['yhat'], df.iloc[training_len:, :]["Close"]))
rmse = round(rmse, 2)

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(df.iloc[training_len:, :]["Close"], forecast['yhat'])

# Calculate MAE
mae = mean_absolute_error(df.iloc[training_len:, :]["Close"], forecast['yhat'])

# Round the values
mape = round(mape, 2)
mae = round(mae, 2)

st.subheader("Value of Root mean squared error, ")
st.write(f'RMSE = {rmse}')

st.subheader("Value of Mean Absolute Error, ")
st.write(f'MAE = {mae}')

st.subheader("Value of Mean Absolute Percentage error, ")
st.write(f'MAPE = {mape}%')

# Plotting graphs between true and predicted values
st.subheader("Comparing Results")

#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])

#Plot/Create the data for the graph
train = data[:training_len]
valid = data[training_len:]
valid['Predictions'] = forecast["yhat"]

#Visualize the data
print(stock + "-")
fig3 = plt.figure(figsize=(10,4))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
st.pyplot(fig3)



# //////////////////////////////////////////////////////////////////////////////

# Case 2: Small Dataset

start = "2023-01-01"
end = "2023-07-31"

# Convert start and end dates to datetime objects
start_date = datetime.strptime(start, "%Y-%m-%d")
end_date = datetime.strptime(end, "%Y-%m-%d")

# Set up the data reader with Yahoo Finance
yf.pdr_override()

df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
df = df.reset_index()

st.subheader("Data for current year")
st.write(df.head())

# Finding training dataset size and splitting accordingly
training_len_small = math.ceil(0.8 * df.shape[0])
training_len_small

stock_training_small = df.iloc[0:training_len_small, : ]

# here we are visualising of closing price
st.subheader("Closing Price Graph for Current year")
fig5 = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.ylabel('Close')
plt.xlabel(None)
st.pyplot(fig5)

# Select only the important features i.e. the date and price
stock_training_small = stock_training_small[["Date","Close"]] # select Date and Price
# Rename the features: These names are NEEDED for the model fitting
stock_training_small = stock_training_small.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset

m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(stock_training_small) # fit the model using all data

future = m.make_future_dataframe(periods=45) #we need to specify the number of days in future
forecast2 = m.predict(future)

forecast = forecast2[forecast2['ds'].isin(df.iloc[training_len_small:, :]["Date"])]

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.head())

# Calculate all metrics
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(forecast['yhat'], df.iloc[training_len_small:, :]["Close"]))
rmse = round(rmse, 2)

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(df.iloc[training_len_small:, :]["Close"], forecast['yhat'])

# Calculate MAE
mae = mean_absolute_error(df.iloc[training_len_small:, :]["Close"], forecast['yhat'])

# Round the values
mape = round(mape, 2)
mae = round(mae, 2)

st.subheader("Value of Root mean squared error, ")
st.write(f'RMSE = {rmse}')

st.subheader("Value of Mean Absolute Error, ")
st.write(f'MAE = {mae}')

st.subheader("Value of Mean Absolute Percentage error, ")
st.write(f'MAPE = {mape}%')


# Plotting graphs between true and predicted values
st.subheader("Comparing Results")

#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])

#Plot/Create the data for the graph
train = data[:training_len_small]
valid = data[training_len_small:]
valid['Predictions'] = forecast2["yhat"]

#Visualize the data
print(stock + "-")
fig6 = plt.figure(figsize=(10,4))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
st.pyplot(fig6)




# ////////////////////////////////////////////////////////////
# Doing random predictions 
# Predicting from user input data
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

askedFuture = m.make_future_dataframe(periods=period)
askedForecast = m.predict(askedFuture)


# Show and plot forecast
st.subheader('Forecast data')
st.write(askedForecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)