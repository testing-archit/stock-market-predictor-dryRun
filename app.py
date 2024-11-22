from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
from datetime import datetime, timedelta
import os
import json
from typing import Tuple, List
import tensorflow as tf
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    MODEL_FOLDER = 'saved_models'
    SCALER_FOLDER = 'saved_scalers'
    SEQUENCE_LENGTH = 60
    FUTURE_DAYS = 30
    TRAINING_YEARS = 2
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.1
    VALID_TICKERS_PATTERN = r'^[A-Z]{1,5}$'

# Create directories
os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
os.makedirs(Config.SCALER_FOLDER, exist_ok=True)

def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol format."""
    if not ticker:
        return False
    return bool(re.match(Config.VALID_TICKERS_PATTERN, ticker))

def get_stock_data(ticker: str) -> np.ndarray:
    """Fetch historical stock data using yfinance."""
    try:
        if not validate_ticker(ticker):
            raise ValueError(f"Invalid ticker format: {ticker}")
            
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{Config.TRAINING_YEARS}y")
        
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        if len(df) < Config.SEQUENCE_LENGTH:
            raise ValueError(f"Insufficient historical data for {ticker}. Need at least {Config.SEQUENCE_LENGTH} days.")
            
        return df['Close'].values
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise

def prepare_data(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare data for LSTM model.
    
    Args:
        data (np.ndarray): Raw stock price data
        sequence_length (int): Number of time steps to look back
        
    Returns:
        Tuple containing:
            - X (np.ndarray): Training sequences
            - y (np.ndarray): Target values
            - scaler (MinMaxScaler): Fitted scaler object
    """
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    return np.array(X), np.array(y), scaler

def create_model(sequence_length: int) -> Sequential:
    """Create LSTM model architecture."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def save_model_and_scaler(model: Sequential, scaler: MinMaxScaler, ticker: str) -> None:
    """Save the trained model and scaler."""
    model_path = os.path.join(Config.MODEL_FOLDER, f'{ticker}_model.keras')
    scaler_path = os.path.join(Config.SCALER_FOLDER, f'{ticker}_scaler.pkl')
    
    model.save(model_path)
    pd.to_pickle(scaler, scaler_path)
    logger.info(f"Model and scaler saved for {ticker}")

def load_model_and_scaler(ticker: str) -> Tuple[Sequential, MinMaxScaler]:
    """Load the trained model and scaler."""
    model_path = os.path.join(Config.MODEL_FOLDER, f'{ticker}_model.keras')
    scaler_path = os.path.join(Config.SCALER_FOLDER, f'{ticker}_scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise ValueError("Model or scaler not found. Please train the model first.")
        
    model = load_model(model_path)
    scaler = pd.read_pickle(scaler_path)
    return model, scaler

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint to get future stock price predictions."""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        market = data.get('market', 'US')  # Default to US market if not provided
        
        if not validate_ticker(ticker):
            return jsonify({'error': f'Invalid ticker format: {ticker}. Must be 1-5 capital letters.'}), 400
        
        logger.info(f"Making predictions for ticker: {ticker}")
        
        # Load model and scaler
        model, scaler = load_model_and_scaler(ticker)
        
        # Get recent data for prediction
        stock_data = get_stock_data(ticker)
        recent_data = stock_data[-Config.SEQUENCE_LENGTH:]
        
        # Scale the data
        scaled_data = scaler.transform(recent_data.reshape(-1, 1))
        
        # Make predictions
        predictions = []
        dates = []
        current_sequence = scaled_data.copy()
        current_date = datetime.now()
        
        for i in range(Config.FUTURE_DAYS):
            current_input = current_sequence[-Config.SEQUENCE_LENGTH:].reshape(1, Config.SEQUENCE_LENGTH, 1)
            next_pred = model.predict(current_input, verbose=0)[0]
            predictions.append(float(next_pred[0]))
            
            # Add business days for dates
            while current_date.weekday() >= 5:  # Skip weekends
                current_date += timedelta(days=1)
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
            
            current_sequence = np.vstack([current_sequence, next_pred])
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return jsonify({
            'predictions': predictions.flatten().tolist(),
            'dates': dates,
            'last_price': float(stock_data[-1]),
            'market': market
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Training endpoint to train the LSTM model."""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        
        if not validate_ticker(ticker):
            return jsonify({'error': f'Invalid ticker format: {ticker}. Must be 1-5 capital letters.'}), 400
        
        logger.info(f"Training model for ticker: {ticker}")
        
        # Get stock data
        stock_data = get_stock_data(ticker)
        
        # Prepare data
        X, y, scaler = prepare_data(stock_data, Config.SEQUENCE_LENGTH)
        
        # Create and train the model
        model = create_model(Config.SEQUENCE_LENGTH)
        model.fit(X, y, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, validation_split=Config.VALIDATION_SPLIT)
        
        # Save the model and scaler
        save_model_and_scaler(model, scaler, ticker)
        
        return jsonify({'message': f'Model trained and saved for {ticker}'}), 200
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting stock prediction server...")
    app.run(port=5020, debug=True)