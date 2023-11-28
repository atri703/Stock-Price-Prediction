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

## Results and Comparisons

• For large dataset, the LSTM and FB-Prophet model generally outperformed ARIMA, having lower error values.
• Based on the smallest RMSE, MAE, and MAPE values, LSTM (Case 1) exhibits the best performance on both datasets. Though FB-Prophet performed significantly better than LSTM (Case 2).
• Performance for Netflix and Adobe was difficult for any model on both datasets when looking at individual stocks. This suggests that their time series data are more complex.
• In terms of runtime, FB-Prophet was the fastest, followed by ARIMA and LSTM respectively. Although in general ARIMA requires the least amount of time to train, because we were using Auto-Arima, which must first determine the optimal hyperparameters, ARIMA took longer than FB-Prophet. Lastly, because LSTM is a deep learning model and needs more computing power than other time series models, it took a lot longer to run than the other two models.
In conclusion, though it forecasts only the next day, LSTM Case 1 is more precise. LSTM Case 2 extends prediction time but decreases accuracy. The FB-Prophet model is best for real-world appli-cations due to its competitive precision and computing economy. It is best for realistic stock price forecasting since it can train models in real time.

## Learning Points

One of the most valuable takeaways from this project was the substantial improvement in my data analysis and machine learning skills. As master's students in Data Science, I had a theoretical com-prehension of statistics; however, through this project, I also gained a deeper understanding of its practical implications for time-series forecasting.
In addition, I acquired knowledge of traditional statistical time series models (ARIMA and FB-Prophet). I learned this for this job through a combination of self-study, research, and application. I've read numerous articles on time series models. I also utilized various online public resources, including YouTube, Coursera, and edx. These materials furnished me with relevant practical exam-ples. In the case of ARIMA, I also had to comprehend the application of various hyperparameters (p, d, and q), as well as their meaning and calculation methodologies. I learned the applications how to plot PACF and ACF graphs, as well as what the Augmented Dickey Fuller (ADF) test is.
In addition, I was able to enhance my web development skills through this project. As a student, I took a web development course, but now I must write backend code in PYTHON for seamless inte-gration with my forecasting models. For this I used a popular PYTHON library known as Streamlit, which takes care of both front-end and backend part of the web app (Streamlit Inc., 2023). I read the official documentation for Streamlit, where I discovered how to install and utilize this library in PY-THON.
Though I had previously completed numerous LSTM projects during my course, this was the first time this model was used extensively for time series forecasting. I was able to refresh my knowledge of testing numerous hyperparameters for deep learning. In addition, I honed my Python programming
skills and became proficient with libraries such as pandas, NumPy, scikit-learn, tensorflow, keras, and y_finance.
Lastly, by calculating all error metrics and comparing the results of each model, I was able to gain a deeper comprehension of each metric's pros and cons, as well as which one should be used in which situation. The same was true for each forecasting model; I now have an improved understanding of each model and greater confidence in my ability to use each model, depending on the circumstances.

## Running the web app

a. Start the web application by executing the following command:
    streamlit run name_of_app.py

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
