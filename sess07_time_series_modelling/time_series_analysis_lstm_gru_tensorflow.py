# Python script to demonstrate advanced time series forecasting models
# using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models

#  Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# Set a random seed for reproducibility
np.random.seed(42)


# Function to generate a synthetic time series data (trend + seasonal + noise)
def generate_synthetic_data(n_points=1000):
   time = np.arange(n_points)
   trend = time * 0.05
   seanality = 10 * np.sin(time / 30)
   noise = np.random.normal(scale=2, size=n_points)
   data = trend + seanality + noise
   return pd.DataFrame({'date': pd.date_range(start='2020-01-01', periods=n_points, freq='D'),
                        'value': data})

# Generate synthetic time series data
df = generate_synthetic_data()

# Display the first 10 entries
print(f"The first 10 entries of the time series are: \n{df.head(10)}")

# Plot the synthetic time series data
plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['value'], label='Synthetic Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Synthetic Time Series Data')
plt.legend()
plt.show()

# Split data into train and test sets
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

# Preprocess data for LSTM/GRU models
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_df['value'].values.reshape(-1, 1))
scaled_test = scaler.transform(test_df['value'].values.reshape(-1, 1))

# Create time series generators for LSTM and GRU models
look_back = 30  # Use the last 30 days to predict the next day's value
train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=look_back, batch_size=32)
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length=look_back, batch_size=32)


# Function to build and train LSTM/GRU models
def build_model(model_type='LSTM'):
   model = Sequential()

   if model_type == 'LSTM':
      model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
   elif model_type == 'GRU':
      model.add(GRU(50, activation='relu', input_shape=(look_back, 1)))

   model.add(Dense(1))
   model.compile(optimizer='adam', loss='mean_squared_error')
   return model


# Build and train the LSTM model
lstm_model = build_model('LSTM')
lstm_model.fit(train_generator, epochs=10, verbose=1)

# Build and train the GRU model
gru_model = build_model('GRU')
gru_model.fit(train_generator, epochs=10, verbose=1)

# Make predictions using both LSTM and GRU models
lstm_predictions = lstm_model.predict(test_generator)
gru_predictions = gru_model.predict(test_generator)

# Inverse scale the predictions and actual values to compare in original scale
lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)
gru_predictions_rescaled = scaler.inverse_transform(gru_predictions)
test_actual_rescaled = scaler.inverse_transform(scaled_test[look_back:])

# Evaluate the models
lstm_rmse = np.sqrt(mean_squared_error(test_actual_rescaled, lstm_predictions_rescaled))
gru_rmse = np.sqrt(mean_squared_error(test_actual_rescaled, gru_predictions_rescaled))

lstm_mae = mean_absolute_error(test_actual_rescaled, lstm_predictions_rescaled)
gru_mae = mean_absolute_error(test_actual_rescaled, gru_predictions_rescaled)

print(f"LSTM RMSE: {lstm_rmse:.4f}")
print(f"GRU RMSE: {gru_rmse:.4f}")
print(f"LSTM MAE: {lstm_mae:.4f}")
print(f"GRU MAE: {gru_mae:.4f}")

# Plot the results
plt.figure(figsize=(12, 5))
plt.plot(df['date'][train_size + look_back:], test_actual_rescaled, label='Actual Data')
plt.plot(df['date'][train_size + look_back:], lstm_predictions_rescaled, label='LSTM Predictions', linestyle='--')
plt.plot(df['date'][train_size + look_back:], gru_predictions_rescaled, label='GRU Predictions', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('LSTM vs GRU Model Predictions')
plt.legend()
plt.show()
