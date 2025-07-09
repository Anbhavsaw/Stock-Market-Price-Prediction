import yfinance as yf                #to fetch data from Yahoo Finance
import pandas as pd                  #for data manipulation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split         # to split data into train and test sets
from sklearn.linear_model import LinearRegression            # to create a linear regression model
from sklearn.metrics import mean_squared_error               # to evaluate the model performance



# Download historical stock data for TCS from Yahoo Finance
data = yf.download('TCS.NS', start='2020-01-01', end='2024-12-31')
print(data.head())                                          # Show first 5 rows


# Using only 'Close' price for prediction
data = data[['Close']]

# Shift the 'Close' price to create 'Next Day Close' column    {basically to form  a target variable for prediction}
data['Next_Close'] = data['Close'].shift(-1)

# Remove last row with NaN value in 'Next_Close'    {Since the last row won’t have a next day’s close value (because no future data exists)}
data = data[:-1]

print(data.head())


######## Split data into training and testing sets ########
 # Features (X) = Today's closing price
X = data[['Close']]

# Labels (y) = Next day's closing price
y = data['Next_Close']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


######## Create and train the linear regression model ########
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)


# Predict on test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Plot actual vs predicted prices
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Price', color='blue', marker='o')              # plot actual prices(test data)
plt.plot(y_pred, label='Predicted Price', color='red', marker='x')                    # plot predicted prices   (model output)
plt.title("TCS Stock Price Prediction (Linear Regression)")                 
plt.xlabel("Days")
plt.ylabel("Stock Price (INR)")
plt.legend()
plt.grid(True)
plt.show()
