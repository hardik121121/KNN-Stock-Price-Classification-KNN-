# 📊 Stock Price Prediction App using KNN 🧑‍💻

Welcome to the **Stock Price Prediction App**! This application uses the **K-Nearest Neighbors (KNN)** algorithm to predict whether a stock will show positive progress (Buy 📈) or negative progress (Sell 📉) based on historical stock data.

## 💡 About
This app fetches stock data from Yahoo Finance using the `yfinance` library, then applies a trained **KNN model** to classify the stock movement as either "Buy" or "Sell" based on the historical closing price. It allows you to input any publicly traded stock ticker, and the model will predict whether the stock is a good buy or sell based on the past performance.

### How It Works:
1. **Input** the stock ticker symbol of any publicly traded company.
2. The app will **fetch historical stock data** from Yahoo Finance for the given ticker.
3. The **KNN model** will classify the stock as either a "Buy" (📈) or "Sell" (📉).
4. The app will display the **accuracy score** of the model and show **actual vs predicted classifications**.

---

## 🚀 Features:
- **User-friendly interface**: Clean and interactive UI for easy stock predictions.
- **Real-time predictions**: Predictions are made using the KNN model based on live stock data.
- **Visualized results**: Accuracy scores and predictions are shown in a clear table format.
- **Customizable**: You can enter any stock ticker and see the model's prediction.
- **Built with heart** 💖 by Hardik Arora.

---

## 🛠️ Technologies Used:
- **Python**: Main programming language for data processing and model training.
- **Streamlit**: Web framework for building the app’s frontend.
- **K-Nearest Neighbors (KNN)**: Classification algorithm used for predictions.
- **yfinance**: Fetching historical stock data from Yahoo Finance.
- **Joblib**: Saving and loading the trained KNN model.

---

## 🏁 How to Run Locally:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction-app.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd stock-price-prediction-app
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run stock_prediction_app.py
   ```

---

## 📈 Example Input:

### Stock Ticker Example: `TATACONSUM.NS`

- **Input**: `TATACONSUM.NS`
- **Prediction**: The model will predict whether the stock should be bought 📈 or sold 📉 based on its past performance.

---

## 🔧 How to Contribute:

1. Fork the repo.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new Pull Request.

---

## 💖 Made with **heart** by Hardik Arora

### Thank you for using this app! Feel free to give feedback or contribute to improving the model! 😊
