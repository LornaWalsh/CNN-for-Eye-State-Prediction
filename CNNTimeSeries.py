import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Train/Test 
from sklearn.model_selection import train_test_split
# Network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,Input,Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Raw data
rawdata = pd.read_csv('EEG.csv',header=0) #Make a dynamic link

# Make working copy
eeg_data = rawdata.copy() 

# Check for erroneous data types
print(eeg_data.dtypes) 

# Visual check for outliers
eeg_in = eeg_data.iloc[:,0:13] # remove output 
print(eeg_in.shape[1])
eeg_in.plot()
plt.show()


# Filter rows based on column IQR values. 
def removeOutliers(data, col):
    Q1 = np.quantile(data[col], 0.25)
    Q3 = np.quantile(data[col], 0.75)
    IQR = Q3 - Q1
    # Set boundaries 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter data based on rows with cell values that are all within range
    in_range = [x for x in data[col] if ((x > lower_bound) & (x < upper_bound))]
    filtered_data = data.loc[data[col].isin(in_range)]
 
# Call function to remove outliers 
for i in eeg_data.columns:
    if i == eeg_data.columns[0]:
        removeOutliers(eeg_data, i)
    else:
        removeOutliers(filtered_data, i)

# Working dataset
eeg_clean = filtered_data

# Visual check that data is clean
eeg_clean.plot()
plt.show()
eeg_clean.isnull().sum()


# Normalise data
# Function for min-max scaling
def min_max_scaling(data):
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())       
    return data

# Call function to normalise data
eeg_norm = min_max_scaling(eeg_clean)
eeg_norm.plot()
plt.show()

# Split into train/validate/test sets
# Remove target vairiable and store in seperate array
X_train = eeg_norm.drop('result',axis=1)
Y_train = eeg_norm.result

# Testing Data - 10%
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.1, stratify=eeg_norm['result'], random_state = 42)


# Define CNN archiecture
batch_size = 64
epochs = 20
num_classes = 2


# define model
eeg_cnn = Sequential()
eeg_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(14, 1)))
eeg_cnn.add(MaxPool1D(pool_size=2))
eeg_cnn.add(Flatten())
eeg_cnn.add(Dense(50, activation='relu'))
eeg_cnn.add(Dense(1))
eeg_cnn.compile(optimizer='adam', loss='mse')


#compile network
eeg_cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#print summary
print(eeg_cnn.summary())

# Fit network
eeg_cnn.fit(x_train, y_train, epochs=1000, verbose=0)

# Analyise training perdictions
train_predict = eeg_cnn.predict(x_train, verbose=0)
accuracy_score(y_train, train_predict)

# Run predictions on test set 
test_predict = eeg_cnn.predict(x_test, verbose=0)
accuracy_score(y_test, test_predict)

