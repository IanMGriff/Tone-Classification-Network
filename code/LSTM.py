""" LSTM MODEL STUFF """

import numpy as np
import scipy.io as sio
import json
import tensorflow as tf
from pandas import DataFrame, Series, concat
from tensorflow.python.keras.layers import Input, Dense, LSTM
from tensorflow.python.keras.models import Sequential
from random import randrange
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


#-------------------------------------------------------------------------------
#  Set keras modules to variables; define helper functions for data processing.
# these functions are modified (slightly) from this tutorial:
    # https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


#==============================================================================



jsonFile = "tone_HbO_HbR__dict_40_sec"

with open('../data/' + jsonFile + '.json') as data_f:
         data = json.load(data_f)



# Load data from selected tone. Tone 1 in this case
data = np.concatenate([np.array(data['0'][ix]) for ix in range(1, len(data['0']))])

X = data.reshape(-1, data.shape[-1])
X = X[:,20]
X = X.reshape(X.shape[0], 1)
encoder = LabelEncoder()

Xi = np.array([encoder.fit_transform(X[:,i]) for i in range(X.shape[1])])
Xi = Xi.T
Xi = Xi.astype("float32")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(Xi)

n_secs = 20
n_features = 1

reframed = series_to_supervised(scaled, n_secs, 1)

train_shape = int(data.shape[1] * (data.shape[0]/14 * 4))  # timesteps * number of rows for 13 participants
values = reframed.values
train = values[:train_shape, :]
test = values[train_shape:, :]

# split into input and outputs
n_obs = n_secs * n_features
train_X, train_y = train[:, :n_obs], train[:,-n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]

train_X = train_X.reshape((train_X.shape[0], n_secs, n_features))
test_X = test_X.reshape((test_X.shape[0], n_secs, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_secs*n_features))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)































## Junk code to save for later (maybe)

#-------------------------------------------------------------------------------
# Feature extraction Following refs.
# Taking mean, variance, kurtosis, and skewness
# of timesteps in the kth channel for the ith subject
#
# mean_dict = {}
# for label, features in data.items():
# 	mean_dict[label] = np.mean(features, axis=1)								# Get mean of each fNirs Channel
#
# variance_dict = {}
# for label, features in data.items():
# 	variance_dict[label] = np.var(features, axis=1)								# Get variance of each fNirs Channel
#
# kurtosis_dict = {}
# for label, features in data.items():
# 	mean_dict[label] = kurtosis(features, axis=1)								# Get kurtosis of each fNirs Channel
#
# skewness_dict = {}
# for label, features in data.items():
# 	mean_dict[label] = skew(features, axis=1)									# Get skewness of each fNirs Channel
#


# -------------------------------------------------------------------------------
# Create Train and Test sets of data (X) and labels (y).
# Concatenate tone info into one matrix for data, mapped row-wise to label matrix.

	 # Select Feature method
# method_dict = data


# train_shape = data['0'].shape[0] - 9
# val_shape = (int(train_shape * 0.9) + 1) * 3
# Build train sets
# y = Y[:train_shape]
# X_train = X[:train_shape]
#
# # Build test Sets
#
# # y_test = np.concatenate([y_label(ix, data[str(ix)][train_shape:]) for ix in range(len(data.keys()))])
# # X_test = np.concatenate([feature_data[train_shape:,] for feature_data in method_dict.values()])
#
# y_test = Y[train_shape:]
# X_test = X[train_shape:]

## -------------------------------------------------------------------------------
## Old way to scale data

# flat_train = X_train.reshape(-1, X_train.shape[-1])		#flatten to preserve channesl & timesteps
# # flat_test = X_test.reshape(-1, X_test.shape[-1])
#
# scaler = preprocessing.StandardScaler().fit(flat_train)
#
# scaled_train = scaler.transform(flat_train)
# # scaled_test = scaler.transform(flat_test)
#
# scaled_train = scaled_train.reshape(X_train.shape)   # return to origninal 3d shape.
# # scaled_test = scaled_test.reshape(X_test.shape)
