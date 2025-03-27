# Stock Forecasting with Multi-Horizon Deep Learning Models

## Overview
This project demonstrates an end-to-end deep learning pipeline for forecasting the SPY ETF (S&P 500 index) over multiple horizons. The project integrates data from multiple sources—including price, volume, technical indicators, simulated news sentiment, and simulated fundamental (macroeconomic) data—to predict multi-day log returns. In addition, a backtesting framework is implemented to evaluate a trading strategy based on the forecasts.

This repository showcases a variety of model architectures (LSTM, Transformer, CNN-LSTM, GRU, Bidirectional LSTM, TCN) and ensemble techniques to demonstrate a breadth of deep learning approaches for time series forecasting.

## Table of Contents
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Training & Hyperparameter Tuning](#training--hyperparameter-tuning)
- [Multi-Horizon Forecasting](#multi-horizon-forecasting)
- [Ensemble & Comparison](#ensemble--comparison)
- [Backtesting & Strategy](#backtesting--strategy)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)
- [Author](#author)


## Data
- **SPY Data:** Downloaded using the [yfinance](https://pypi.org/project/yfinance/) API covering the period from 2000 to 2025.
- **VIX Data:** Downloaded to capture market volatility.
- **Simulated News Sentiment:** Daily sentiment scores (values between -1 and 1) to represent market sentiment.
- **Simulated Fundamentals:** Monthly data for Inflation and Interest Rate; these values are forward-filled to create daily features.
- **Technical Indicators:** Calculated indicators include moving averages (50-day, 200-day), RSI, MACD, Bollinger Bands, and historical volatility.

## Feature Engineering
- **Technical Indicators:**  
  - **Moving Averages:** Smooth price data and capture trend direction.
  - **RSI:** Measures momentum and potential overbought/oversold conditions.
  - **MACD & Signal Line:** Used to identify trend reversals.
  - **Bollinger Bands:** Provide upper and lower bounds of price volatility.
  - **Historical Volatility:** Annualized standard deviation of daily log returns over a 20-day window.
- **Volume & Sentiment:**  
  - Volume trends (Volume moving average, percentage change).
  - Simulated news sentiment captures qualitative market signals.
- **Fundamental Data:**  
  - Macro indicators like Inflation and Interest Rate, which are important for broad market indices like SPY.
- **Additional Data:**  
  - VIX as a proxy for overall market volatility.

## Data Preparation
- **Windowing:**  
  A sliding window approach (e.g., 30 days) is used to capture temporal dependencies.
- **Multi-Horizon Targets:**  
  Targets are defined as multi-day log returns (e.g., 1-day, 3-day, 5-day log returns) computed as:
  \[
  \text{LogReturn}_{t+h} = \ln\left(\frac{P_{t+h}}{P_t}\right)
  \]
- **Scaling:**  
  Both features and targets are scaled using `StandardScaler` to improve model training.

## Model Architectures
The project implements several deep learning architectures:
- **Simple LSTM**  
- **Transformer (Multi-Head Attention)**  
- **CNN-LSTM**  
- **GRU**  
- **Bidirectional LSTM**  
- **TCN (Temporal Convolutional Network)**  
- **Multi-Horizon Forecasting Model:**  
  Each model is adapted to output a vector of predictions for different forecast horizons.

## Training & Hyperparameter Tuning
- **Training Setup:**  
  Models are trained on an 80/20 train-test split using callbacks such as EarlyStopping and ReduceLROnPlateau.
- **Hyperparameter Tuning:**  
  Grid search over learning rates, dropout rates, number of attention heads, and feed-forward dimensions was performed to optimize model performance.
- **Results:**  
  Validation loss and RMSE metrics (on multi-day and one-step-ahead log returns) are used to compare models.

## Multi-Horizon Forecasting
- The target for each sample is a vector (e.g., `[return_1d, return_3d, return_5d]`), enabling the model to capture different time scales in its predictions.

## Ensemble & Comparison
- **Model Ensemble:**  
  Predictions from multiple architectures are averaged to form an ensemble prediction.
- **Comparison:**  
  RMSE is computed for each forecast horizon for individual models and for the ensemble, demonstrating the strengths of combining models.

## Backtesting & Strategy
- **Trading Strategy:**  
  A simple backtesting framework was implemented based on the model’s one-day forecast (or a combination of horizons). The strategy uses threshold-based signals to decide long/short positions, includes rudimentary stop-loss/take-profit rules, and factors in transaction costs.
- **Performance Metrics:**  
  The cumulative returns of the strategy are compared to a buy-and-hold baseline, and key metrics (final returns, RMSE, etc.) are reported.

## Results
- **Forecasting Performance:**  
  - RMSE on multi-day log returns is typically around 0.027–0.028.
  - One-step-ahead RMSE on daily log returns is around 0.013–0.014.
- **Backtesting:**  
  The basic signal-based trading strategy underperforms buy-and-hold on a bullish index like SPY, but further refinements (e.g., position sizing, risk management) are suggested for future work.

## Usage
- Clone the repository:
   ```bash
   git clone https://github.com/SeanMZ7/stock-forecasting-project-1

## Future Work
- Explore ensemble methods (weighted averaging, stacking) to further improve prediction accuracy.
- Integrate additional high-quality sentiment data and more comprehensive macroeconomic indicators (e.g., GDP growth, unemployment rates).
- Experiment with more advanced architectures (e.g., deeper Transformers, attention-based models) and risk management strategies.
- Implement walk-forward validation for more robust performance evaluation.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.

## Author
- Sean Ziogas
- SeanMZ7
