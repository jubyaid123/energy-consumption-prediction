# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# %%
data = pd.read_excel('data/shepard_energy_consumption.xlsx')
data.rename(columns={'CUNY City College of New York - 2018-09-01 -> 2023-08-31': 'Timestamp'}, inplace=True)
data.drop(data.index[0], inplace=True)
data.rename(columns={'12785711': 'Meter1'}, inplace=True)
data.rename(columns={'12785712': 'Meter2'}, inplace=True)



# %%
data.head()

# %%
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)



# %%
data.head()

# %%
# Nomalize data for meter1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Meter1'].values.reshape(-1, 1))

# %%
print(scaled_data[0:25])
scaled_data[24]

# %%
# Function to create dataset for LSTM
# stores values in array X that will be used to predict value in correspending index in Y
# first iteration
# 0-23 values stored in X, 24th value stored in Y(target value)
# second iteration 
# 1-24 stored in X, 25th value stored in Y
def create_dataset(dataset, look_back=24):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1): #ensures that loop stops at a point where there is enough datapoints for a last X, and Y set
        a = dataset[i:(i + look_back), 0]  #slices values in dataset from i to i+24
        X.append(a)
        Y.append(dataset[i + look_back, 0]) # stores the value right after a into Y
    return np.array(X), np.array(Y)

# %%
# Creating the dataset suitable for LSTM
X, Y = create_dataset(scaled_data)

# %%
print(X[0])
print(Y[0])

# %%
print(X.shape[1])


# %%


# Reshaping input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Splitting the dataset into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# %%
X.shape

# %%

# Building the LSTM network
## sequential model means layers are a linear stack where layers can be added one after another 
model = Sequential()
## first layer has 50 neurons, inpputshape(time steps, features) return_sequences because there is another layer
model.add(LSTM(50, input_shape=(X_train.shape[1], 1), return_sequences=True))
# second and last layer 
model.add(LSTM(50))
# output for final predictin Dense means the neuron is fully connected with previous neurons
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')


# %%
model.summary()

# %%
# Training the model
model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=1)
#model starts processing data in X_train and amkes predictions for y_train adjusts every epoch based on MSE thus training it 

# %%
train_predict = model.predict(X_train)
# model makes prediction for energy output based on previously seen data
#train_predict should predict the valyes in Y_train
test_predict = model.predict(X_test)
# model makes predictions based of new data
# test_predict should predict the valyes in Y_test

# %%

# Inverting predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train_inv = scaler.inverse_transform([Y_train])
Y_test_inv = scaler.inverse_transform([Y_test])

# %%
Y_train_inv.shape
print(Y_train_inv[0])


# %%
# Calculating RMSE
train_rmse = np.sqrt(mean_squared_error(Y_train_inv[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(Y_test_inv[0], test_predict[:,0]))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# %%
# Plotting the results
plt.figure(figsize=(14, 6))
plt.plot(data['Meter1'][-len(Y_test):].index, Y_test_inv[0], label='Actual')
plt.plot(data['Meter1'][-len(Y_test):].index, test_predict[:, 0], label='Predicted')
plt.title('Meter Energy Output Prediction')
plt.xlabel('Date')
plt.ylabel('Energy Output')
plt.legend()
plt.show()

# %%
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(data['Meter1'][-len(Y_test):].index, test_predict[:, 0], label='Predicted')
plt.title('Meter Energy Output Prediction')
plt.xlabel('Date')
plt.ylabel('Energy Output')
plt.legend()
plt.show()

# %%
recent_data_week = data['Meter1'][-24:]
scaled_recent_data_week = scaler.transform(recent_data_week.values.reshape(-1,1))
print(scaled_recent_data_week)

# %%
sequence_week = np.reshape(scaled_recent_data_week, (1, 24, 1))
predictions_week = []

# %%
for i in range(7*24):  # Loop for 7 days 24 hours a day 
    daily_prediction_scaled = model.predict(sequence_week)
    daily_prediction = scaler.inverse_transform(daily_prediction_scaled)
    predictions_week.append(daily_prediction[0, 0])
    
    # Update the sequence: roll and append the new prediction
    sequence_week = np.roll(sequence_week, -1)
    sequence_week[0, -1, 0] = daily_prediction_scaled[0, 0]

# %%
print(len(predictions_week))
print(predictions_week)


# %%
# Start and end dates
start_date = pd.to_datetime('2023-09-01 01:00')
end_date = pd.to_datetime('2023-09-08 00:00')  

# Generate a list of hourly timestamps
timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

# %%
print(len(timestamps))
print(len(predictions_week))


# %%
plt.figure(figsize=(15, 5))  # Adjust the size as needed
plt.plot(timestamps, predictions_week, label='Predicted Energy Output')
plt.xlabel('Timestamp')
plt.ylabel('Energy Output')
plt.title('Hourly Energy Output Predictions from 2023-09-01 to 2023-09-07')
plt.xticks(rotation=45)  # Rotates the X-axis labels for readability
plt.legend()
plt.show()

# %%
recent_data_month = data['Meter1'][-24:]
scaled_recent_data_month = scaler.transform(recent_data_month.values.reshape(-1,1))
sequence_month = np.reshape(scaled_recent_data_month, (1,24,1))
print(scaled_recent_data_month)

# %%
predictions_month = []
for i in range(30 * 24):  # 30 days * 24 hours
    hourly_prediction_scaled = model.predict(sequence_month)
    hourly_prediction = scaler.inverse_transform(hourly_prediction_scaled)
    predictions_month.append(hourly_prediction[0, 0])

    sequence_month = np.roll(sequence_month, -1)
    sequence_month[0, -1, 0] = hourly_prediction_scaled[0, 0]


# %%
start_date_month = pd.to_datetime('2023-09-01 00:00')
end_date_month = pd.to_datetime('2023-09-30 23:00')
timestamps_month = pd.date_range(start=start_date_month, end=end_date_month, freq='H')


# %%
print(len(timestamps_month))
print(len(predictions_month))


# %%
print(predictions_month)

# %%
plt.figure(figsize=(15, 5))
plt.plot(timestamps_month, predictions_month, label='Predicted Energy Output')
plt.xlabel('Timestamp')
plt.ylabel('Energy Output')
plt.title('Hourly Energy Output Predictions for September 2023')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# %%
import csv

# %%
with open('week_predictions.csv', 'w', newline='')as file:
    writer = csv.writer(file)
    for prediction in predictions_week:
        writer.writerow([prediction])


