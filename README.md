# Stock-Price-Prediction
Explore stock price forecasting using traditional ARIMA, modern LSTM-based RNN, and FB-Prophet models. The web app showcases their performance for easy access by both technical and non-technical users.

## Overview

This project aims to predict stock prices using a combination of traditional time-series models (ARIMA), modern deep learning techniques (LSTM-based RNN), and Facebook Prophet. The goal is to provide accurate forecasts and create an interactive web application for users to visualize and compare predictions.

## Scope
The scope of this project includes the development, evaluation, and comparison of time series mod-els for forecasting of stock prices and trends. It will also undersee, the applications of all models namely, ARIMA, Facebook Prophet, and LSTM model. Furthermore, this dissertation explores two LSTM cases, one designed for short-term forecasting with a one-day prediction horizon and another for longer-term predictions spanning multiple days. The evaluation phase includes the analysis of key performance indicators that are, RMSE, MAE and, MAPE.
The Scope further expanded to build an interactive web application using Streamlit for user-friendly stock price predictions analysis. It is important to note that this project does not consider the qualita-tive analysis, sentimental analysis, cooperate management factors, public perceptions, or any other miscellaneous variables that can impact stock prices, because focus of this project is to assess the performance of each model through quantitative analysis of historical stock price data obtained from Yahoo Finance and using it for predictive modelling.

## Approach
In this project we will adopt a data-driven approach to tackle the forecasting challenges analysts and investors face in the financial markets. Our methodology primarily entails the implementation and comparison of three forecasting models i.e., ARIMA, FB-Prophet and LSTM. To achieve this, we will utilize Yahoo Finance as the data source to obtain historical stock price data. Furthermore, our pro-ject involves the development of an interactive web application using Streamlit package to have a user-friendly access to our forecasting tools. This multimodal approach aims to provide accurate, precise, and timely stock price predictions (or trends) while ensuring usability for a wide range of audience.

## Key Features

- **Multi-Model Prediction:** Utilizes ARIMA, LSTM, and Prophet models for comprehensive stock price forecasting.
- **Web Application:** An interactive web app allows users to input parameters, visualize predictions, and compare different models.
- **Data Collection:** Historical stock price data is obtained from Yahoo Finance using the yfinance Python library.
- **Two Dataset Cases:** The project considers both large and small datasets to assess model performance under varying data sizes.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries (pandas, scikit-learn, TensorFlow, yfinance)
- Web browser

## Usage

Input parameters in the web application.
Visualise historical and predicted stock prices.
Compare predictions from different models.

## Models

### ARIMA Model

Traditional time-series model capturing linear relationships in stock price data.

### LSTM Model

Deep learning model (RNN) capturing complex patterns and dependencies in sequential data.

### Prophet Model

Open-source forecasting tool designed for time-series data, robust in handling seasonality and trends.

## Data Collection

In each model, data was obtained from the yfinance library. As indicated previously, two samples were collected for each stock to improve analysis. Intriguingly, we were unable to integrate 2007 stock data for X (Twitter) and Meta (Facebook) due to a name change; thus, to maintain consistency, we conducted research on the remaining eight models. Furthermore, there are two different types of stock for Google i.e., Type A & Type C. Out of these two we used data of Type C because there is influence on price of share due to company’s internal decision (Yahoo Inc., 2023).
Other libraries that were used are:
• datetime – To easily separate year, month and day of the given date
• pandas_datareader.data.get_data_yahoo – To swiftly import data from yfinance and inte-grate it with the pandas library.

## Evaluation

Model performance is evaluated using metrics like RMSE, MAPE, and MAE. The results section provides a detailed comparison of the forecasting models.

## Results

The project's results highlight strengths and weaknesses, providing insights into the effectiveness of each forecasting model.

## Learning Points

Key learnings include skills acquired, crucial actions for success, and considerations for future work.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
