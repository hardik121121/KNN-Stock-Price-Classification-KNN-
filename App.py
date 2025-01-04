import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the pre-trained KNN model
model = joblib.load('knn_model.pkl')

# Function to fetch stock data
def get_stock_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2024-12-31")
    data['Open - Close'] = data['Open'] - data['Close']
    data['High - Low'] = data['High'] - data['Low']
    data = data.dropna()
    return data

# Streamlit app title and emojis for a friendly touch
st.title("📊 Stock Price Prediction 🧑‍💻")

# Description of the app
st.write("""
This app uses K-Nearest Neighbors (KNN) algorithm to predict whether a stock will show positive progress (Buy 📈) or negative progress (Sell 📉) based on historical stock data.

You can enter any stock ticker (e.g., `TATACONSUM.NS`, `AAPL`, etc.) to get predictions on future stock movement!

### How it works:
1. Input the stock ticker symbol of any publicly traded company.
2. The app fetches historical stock data for the given ticker.
3. The KNN model classifies the stock as a "Buy" or "Sell" based on its price behavior.
4. See the model's prediction accuracy and results displayed.

Let's get started! 😊
""")

# User input for stock ticker with a placeholder text
ticker_input = st.text_input("Enter Stock Ticker (e.g., TATACONSUM.NS):", value="TATACONSUM.NS")

if ticker_input:
    # Fetch stock data based on user input
    st.write(f"🔍 Fetching stock data for **{ticker_input}**")
    data = get_stock_data(ticker_input)
    
    # Prepare features and target for classification
    X = data[['Open - Close', 'High - Low']]
    Y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=44)
    
    # Reshape target variables
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()
    
    # Make predictions using the loaded model
    predictions_classification = model.predict(X_test)
    
    # Display results
    accuracy_test = accuracy_score(Y_test, predictions_classification)
    
    # Show accuracy score with emoji
    st.write(f"✅ **Test Accuracy:** {accuracy_test * 100:.2f}%")
    
    # Display the predictions in a visually appealing way
    actual_predicted_data = pd.DataFrame({'Actual Class': Y_test, 'Predicted Class': predictions_classification})
    
    st.write("📊 **Actual vs Predicted Classes**")
    st.write(actual_predicted_data.head(10))
else:
    st.write("🚀 Please enter a valid stock ticker symbol to get started! 📈📉")

# Heart message at the end
st.markdown("""
---
💖 Made with **heart** by - **Hardik Arora** 💻
""")

