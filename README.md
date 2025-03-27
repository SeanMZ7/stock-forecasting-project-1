# Stock Forecasting with Multi-Horizon Deep Learning Models

## Overview
This project forecasts the SPY ETF (S&P 500 index) using a range of advanced deep learning models (LSTM, Transformer, CNN-LSTM, GRU, Bidirectional LSTM, TCN) along with engineered features from price, volume, technical indicators, news sentiment, and fundamental macroeconomic data. It also includes a backtesting framework to simulate a trading strategy based on the forecasts.

## Data
- **Source:** Data is downloaded using the yfinance API for SPY and VIX.
- **Additional Data:** Simulated daily news sentiment and monthly fundamental data (Inflation, Interest Rate) are merged with the stock data.
- **Technical Indicators:** Moving averages, RSI, MACD, Bollinger Bands, and historical volatility are computed.

## Methods
- **Data Preparation:**  
  - Features are scaled and sequences are constructed using a sliding window (window size = 30 days).
  - Multi-horizon targets are computed as log returns over forecast horizons (e.g., 1-day, 3-day, 5-day).
- **Model Architectures:**  
  - Multiple deep learning models are implemented, including LSTM, Transformer, CNN-LSTM, GRU, Bidirectional LSTM, and TCN.
- **Ensembling:**  
  - Predictions from various models are combined using averaging to improve robustness.
- **Backtesting:**  
  - A simple trading strategy is backtested using model signals with threshold-based long/short decisions and risk management rules.

## Results
- **Forecasting:** RMSE metrics are computed for each forecast horizon.
- **Backtesting:** Cumulative returns from the model-based strategy are compared with a buy-and-hold strategy.
- Detailed plots and performance metrics are provided in the notebook.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/SeanMZ7/stock-forecasting-project-1
