from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = 'stock_prediction_model.keras'
SCALER_PATH = 'stock_scaler.pkl'

def format_symbol(ticker, market='US'):
    if market == 'BSE':
        return f"{ticker}.BO"
    elif market == 'NSE':
        return f"{ticker}.NS"
    return ticker

def load_data(ticker, market='US', start_date=None, end_date=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    formatted_ticker = format_symbol(ticker, market)
    try:
        data = yf.download(formatted_ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker {formatted_ticker}")
        if len(data) < 60:
            raise ValueError(f"Insufficient historical data for {formatted_ticker}")
        
        # Reset index to make date accessible as a column
        data = data.reset_index()
        data.ffill(inplace=True)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        raise Exception(f"Error fetching data for {formatted_ticker}: {str(e)}")

def compute_technical_indicators(data):
    df = data.copy()
    
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2*bb_std
    df['BB_lower'] = df['BB_middle'] - 2*bb_std
    
    # Daily Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Drop any rows with NaN values
    df.dropna(inplace=True)
    
    return df

def prepare_sequences(data, sequence_length=60):
    # Select features for prediction
    feature_columns = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Returns', 'Volatility']
    features = data[feature_columns].values
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    X = []
    y = []
    
    # Create sequences
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:(i + sequence_length)])
        y.append(scaled_features[i + sequence_length, 0])  # 0 index for Close price
    
    return np.array(X), np.array(y), scaler

def create_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model

def train_model(ticker, market='US', start_date=None, end_date=None):
    try:
        # Load and prepare data
        data = load_data(ticker, market, start_date, end_date)
        data = compute_technical_indicators(data)
        
        # Prepare sequences and get scaler
        X, y, scaler = prepare_sequences(data)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Create and train model
        model = create_model((X.shape[1], X.shape[2]))
        
        checkpoint = ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=0
        )
        
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint],
            verbose=1
        )
        
        # Save scaler
        pd.to_pickle(scaler, SCALER_PATH)
        
        return model, scaler
        
    except Exception as e:
        raise Exception(f"Error during model training: {str(e)}")

def predict_future(model, scaler, last_sequence, prediction_days=30):
    current_sequence = last_sequence.copy()
    predictions = []
    
    for _ in range(prediction_days):
        # Predict next value
        current_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), verbose=0)
        predictions.append(current_pred[0, 0])
        
        # Update sequence
        new_row = current_sequence[-1].copy()
        new_row[0] = current_pred[0, 0]  # Update Close price
        
        # Update other features (simplified)
        new_row[2] = np.mean(current_sequence[-5:, 0])  # MA5
        new_row[3] = np.mean(current_sequence[-20:, 0])  # MA20
        new_row[8] = (new_row[0] - current_sequence[-1, 0]) / current_sequence[-1, 0]  # Returns
        
        # Shift sequence and add new prediction
        current_sequence = np.vstack((current_sequence[1:], new_row))
    
    # Inverse transform predictions
    dummy_array = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy_array[:, 0] = predictions
    predictions_rescaled = scaler.inverse_transform(dummy_array)[:, 0]
    
    return predictions_rescaled

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        market = data.get('market', 'US')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required!'}), 400
        
        print(f"Training model for {ticker} in {market} market...")
        model, _ = train_model(ticker, market, start_date, end_date)
        
        return jsonify({
            'message': f'Model successfully trained for {ticker}',
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        market = data.get('market', 'US')
        prediction_days = int(data.get('prediction_days', 30))
        
        if not ticker:
            return jsonify({'error': 'Ticker symbol is required!'}), 400
        
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({'error': 'Model not trained. Please train the model first.'}), 400
            
        # Load model and scaler
        model = load_model(MODEL_PATH)
        scaler = pd.read_pickle(SCALER_PATH)
        
        # Get recent data
        data = load_data(ticker, market)
        data = compute_technical_indicators(data)
        
        # Prepare features
        feature_columns = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Returns', 'Volatility']
        features = data[feature_columns].values
        scaled_features = scaler.transform(features)
        last_sequence = scaled_features[-60:]
        
        # Make predictions
        predictions = predict_future(model, scaler, last_sequence, prediction_days)
        
        # Generate future dates
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, prediction_days + 1)]
        future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        # Calculate confidence intervals
        volatility = data['Volatility'].iloc[-1]
        upper_bound = predictions * (1 + volatility * 2)
        lower_bound = predictions * (1 - volatility * 2)
        
        response_data = {
            'predictions': predictions.tolist(),
            'upper_bound': upper_bound.tolist(),
            'lower_bound': lower_bound.tolist(),
            'dates': future_dates,
            'last_actual_price': float(data['Close'].iloc[-1]),
            'ticker': ticker,
            'market': market
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5020, debug=True)