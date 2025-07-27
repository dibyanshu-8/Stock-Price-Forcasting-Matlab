# MATLAB ARIMA Stock Price Forecasting

This repository contains a MATLAB script (`stock_price_forecasting.m`) for performing time series forecasting on stock closing prices using the Autoregressive Integrated Moving Average (ARIMA) model. The script includes data loading, visualization, stationarity testing, model training, forecasting, and performance evaluation.

## Features

-   **Data Loading & Preparation:** Reads stock data from a CSV file, handles missing values, and prepares the 'Close' price series.
-   **Data Visualization:** Plots historical closing prices and their distribution (KDE).
-   **Stationarity Test:** Performs Augmented Dickey-Fuller (ADF) test and visualizes rolling mean/standard deviation to check for stationarity.
-   **Data Transformation:** Applies logarithmic transformation to stabilize variance.
-   **Train/Test Split:** Divides the data into training (90%) and testing (10%) sets.
-   **Optimal ARIMA Parameter Search:** Iteratively searches for the best `(p, d, q)` ARIMA orders based on AIC (Akaike Information Criterion).
-   **ARIMA Model Fitting:** Fits the ARIMA model to the training data.
-   **Stock Price Forecasting:** Generates future forecasts with 95% confidence intervals.
-   **Forecast Visualization:** Plots the training data, actual test data, and predicted stock prices on a log scale.
-   **Performance Metrics:** Calculates Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) on the log-transformed data.
-   **Residual Analysis:** Performs Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots, and Ljung-Box Q-Test on model residuals to check for white noise.

## Getting Started

### Prerequisites

-   MATLAB (with Statistics and Machine Learning Toolbox for `arima`, `estimate`, `forecast`, `adftest`, `lbqtest`, `autocorr`, `parcorr`).

### Data

The script expects a CSV file named `Download Data - STOCK_US_XNAS_ACGL.csv` in the same directory as the MATLAB script. This file should contain at least the following columns:

-   `Date` (in a recognizable date format)
-   `Close` (numerical closing price)

### How to Run

1.  Save the provided MATLAB code as `stock_price_forecasting.m`.
2.  Place your stock data CSV file (`Download Data - STOCK_US_XNAS_ACGL.csv`) in the same directory.
3.  Open MATLAB and navigate to the directory containing the script.
4.  Run the script from the MATLAB command window:
    ```matlab
    stock_price_forecasting
    ```
