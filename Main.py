import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Funz_Utili as fu
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Columns to remove in the column removal function
Column_Remove = [1, 4, 6]

# Days to consider for calculating the mean
D_media = 1

# Number of variables to consider
Num_Var = 13 - len(Column_Remove)

# Days before the prediction period
D_Before = 10

# Days for prediction
D_Prediction = 7

# Number of cycles based on the prediction and mean days
N_Cicles = int(D_Prediction / D_media)

# Fraction of data used for normalization and testing
frac = 0.8

# Dataframe to hold reframed and scaled data for all regions
ARRSD = pd.DataFrame()

# Create ARRSD dataframe with all variables of all regions ordered from t-nin to t+nout
for i in range(0, 20):
    # Create DataFrame for the current region with specified columns removed
    df = fu.Create_DataFrame(Column_Remove, i, 2, 3)

    # Normalize DataFrame values within the specified fraction
    values = fu.Normalization_DataFrame(df, frac)

    index_d = df.index

    # Convert DataFrame to time series matrix with specified days before and after, and remove rows with NaN values
    df = fu.Data_to_time_series_Matrix(values, D_Before, D_Prediction, index_d, True)

    # Keep specified variable (1st in this case) and remove others
    df = fu.Var_Keep(df, 1, D_Before, D_Prediction)

    # Concatenate current DataFrame with ARRSD DataFrame
    ARRSD = pd.concat([ARRSD, df])

# Create target variables for ARRSD DataFrame
ARRSD = fu.Target(ARRSD, Num_Var, D_Before, D_Prediction, N_Cicles)

# Create training and testing datasets for LSTM model
Train_X, Train_Y, Test_X, Test_Y = fu.Create_Train_XY_and_Test_XY(ARRSD, D_Before, D_Prediction, 20, frac, Num_Var)

# Parameters for LSTM model
LSTM_Units = 200
Learning_Rate = 0.00001
Batch_size = 100
Epoch = 100

# Define and compile LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(Train_X.shape[1], Train_X.shape[2])))
model.add(tf.keras.layers.LSTM(LSTM_Units, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
model.add(tf.keras.layers.LSTM(LSTM_Units, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
model.add(tf.keras.layers.Dense(N_Cicles, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=Learning_Rate), loss='mse', metrics=['accuracy'])

# Train the model
history = model.fit(Train_X, Train_Y, Batch_size, Epoch, validation_data=(Test_X, Test_Y), shuffle=False, verbose=2)

