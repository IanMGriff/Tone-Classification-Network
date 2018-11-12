#-------------------------------------------------------------------------------
""" Formatting script for converting data from *.mat or *.nirs files to python
numpy arrays.This packages the  NIRs time signals and Hb concentration signals
in a Python Dict object. This Dict is saved as a Json file to be used as
input for TensorFlow operations. """

#-------------------------------------------------------------------------------
# Import necessary modules for data reading and Neural Network (NN) Construction.
import numpy as np
from random import randrange
from scipy.stats import skew, kurtosis
import scipy.io as sio
import json
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from tensorflow.python.keras.layers import Bidirectional, Dense, Flatten, Dropout, TimeDistributed
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM, GlobalMaxPooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.python import keras
import matplotlib.pyplot as plt



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#-------------------------------------------------------------------------------
#  Set keras modules to variables; define helper functions for data processing.



class AccuracyHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.acc = []
		self.val_acc = []

	def on_epoch_end(self, batch, logs={}):
		self.acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))




# Split a dataset into k folds
def cross_val_split(dataset, folds=1):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split



def y_label(tone, data, cats=3):
	""" Takes in an int as tone, np array as data
		and returns 1 hot matrix of y labels  """
	n = data.shape[0]
	labels = np.zeros([n, cats])
	labels[:,tone] = [1] * n
	return labels


def Scale(items, labels, tone):

	items = items[labels[:,tone]==1]
	flat_items = items.reshape(-1, items.shape[-1])		#flatten to preserve channesl & timesteps
	scaled = preprocessing.StandardScaler().fit_transform(flat_items)
	scaled = scaled.reshape(items.shape)
	return scaled


def ScaleDataset(items, labels, tones):
	for tone in range(tones):
		items[labels[:,tone]==1] = Scale(items,labels,tone)
	return items



def create_2_layer_CNN():
	model = Sequential()
	model.add(Conv1D(nodes, kernel_size=3, strides=1,
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(dropRate))
	#
	model.add(Conv1D(32, 3, activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(dropRate))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	# model.add(BatchNormalization())

	model.add(Dense(128, activation='relu'))
	# model.add(BatchNormalization())

	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])
	return model

def stacked_2_layer_CNN():
	model = Sequential()
	model.add(Conv1D(nodes, kernel_size=3,
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])
	return model



#-------------------------------------------------------------------------------
#  Load Json with tone and dc data.

jsonFile = "tone_HbO_HbR__dict_40_sec"

with open('../data/' + jsonFile + '.json') as data_f:
         data = json.load(data_f)

# data = {key:val[1:2] for key,val in data.items()}
# OLD WAY TO CONCATENATE DATA
# for key, value in data.items():
# 	value = value[1:]
# 	data[key] = np.concatenate(value)
#

# Convert python lists in data to np.arrays for NN models.
#      dict key is tone category; dict value is array of matrices.
#      each matrix is 3dim holding data per participant
#      rows = data; columns = nirs channel; slice = trial of tone category

# cutSize = int(np.array(data['0'][1]).shape[1] / 4)  # divide 20's time int to get 5's dur.[:,cutSize:]

     # range(1, len..) to temp remove fNIRS_004 (not enough data)
	# merge list of matrices into one 3d array, with alternating tones for k fold
	# i = subject, j = time step, k = fNIRS channel

X = np.concatenate([np.array(data[key][ix]) for ix in range(1, len(data['0'])) for key in data.keys()])
y = np.concatenate([y_label(int(j), np.array(data[j][i])) for i in range(1, len(data['0'])) for j in data.keys()])
# X = np.concatenate([np.array(data[key][1]) for key in data.keys()])
# y  = np.concatenate([y_label(int(j), np.array(data[j][1])) for j in data.keys()])
#

# ## Previous lines are pythonic way of the following for loop

# data_1 = []
# labels = []
# for ix in range(1,len(data['0'])):
# 	data_1.append(data['0'][ix]))
# 	data_1.append(data['1'][ix])
# 	data_1.append(data['2'][ix])
# 	labels.append(y_label(0, np.array(data['0'][ix])))
# 	labels.append(y_label(1, np.array(data['1'][ix])))
# 	labels.append(y_label(2, np.array(data['2'][ix])))
# data_1 = np.concatenate(data_1)
# labels = np.concatenate(labels)




# -------------------------------------------------------------------------------

X_scale = X.copy()
#
X_scale = ScaleDataset(X_scale, y, 3)
#
#
# #-------------------------------------------------------------------------------
# 'Knobs' for nodes in perceptron layers

nodes = X_scale.shape[2]  # Number of nodes in hidden layer
in_size = 34 # Number of nodes in input layer (input channels)
num_classes = 3 # Number of nodes in output layer (output featurs)
epochs = 20		# iteration that training stops on.
batch_size = 22  # Number of samples per gradient update.
dropRate= 0.5


#-------------------------------------------------------------------------------
# Create models using tensorflow keras API and scikit-learn

	# First test, simple MLP using scikit-learn api
	# hidden_layer_sizes parameter is (neurons_layer_i, neurons_layer_i+1, ...)
#
# mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(nodes,), max_iter=4000)
#
# mlp.fit(scaled_train, y_train)
# predictions = mlp.predict(X_test)
#
#
# print("Training set score: %f" % mlp.score(scaled_train, y_train))
# print("Test set score: %f" % mlp.score(scaled_test, y_test))
#
# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))



#-------------------------------------------------------------------------------
## Create tensorflow models using keras API

##   Model 2 CNN
# input_shape = X_train.shape[1:]  # channel entries (timesteps), number of channels
#
# model = KerasClassifier(build_fn=create_2_layer_CNN, verbose=0)
#
# 	## Use Grid Search to determine optimal model parameters
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [10, 50, 100]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# # train_shape = int(X.shape[0] * .9)
# y_train = y
#
# lim = 100
# grid_result = grid.fit(X_train[:lim,], y_train[:lim,])
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


	# Create K-fold loop to validate model

# n_fold = 2 # number of participants included in data
#
# kf = KFold(n_splits=n_fold)
# kf.get_n_splits(X)
#
#
input_shape = X.shape[1:]  # channel entries (timesteps), number of channels
#
# test_acc = []
#
# for train_index, test_index in kf.split(X):
#
# 	## Obtain splits
#
# 	# print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = X[train_index], X[test_index]
# 	y_train, y_test = y[train_index], y[test_index]

# -------------------------------------------------------------------------------
# Standardize data for model using sklearn preprocessing API
# This makes sure data has normalized dist.
#     Subtract mean value from feature, divide non-constants by st. dev.

	# flat_train = X_train.reshape(-1, X_train.shape[-1])		#flatten to preserve channesl & timesteps
	# flat_test = X_test.reshape(-1, X_test.shape[-1])

	# scaler = preprocessing.StandardScaler().fit(flat_train)
	#
	# scaled_train = scaler.transform(flat_train)
	# scaled_test = scaler.transform(flat_test)
	#
	# scaled_train = scaled_train.reshape(X_train.shape)   # return to origninal 3d shape.
	# scaled_test = scaled_test.reshape(X_test.shape)


# scaled_train = ScaleDataset(X_train, y_train_val, 3)
	# scaled_test = ScaleDataset(X_test,y_test, 3)
#
# flat_train = X_train.reshape(-1, X_train.shape[-1])#flatten to preserv
# # flat_test = X_test.reshape(-1, X_test.shape[-1])
# #
# #
# scaled = preprocessing.StandardScaler().fit_transform(flat_train)
# scaled_train = scaled.reshape(X_train.shape)
#
# scaled = preprocessing.StandardScaler().fit_transform(flat_test)
# scaled_test = scaled.reshape(X_test.shape)
#
# 	# val_shape = (int(scaled_train.shape[0] * 0.9))
#
trainSize = int(X.shape[0] * .8)

X_train = X_scale[:trainSize]
y_train = y[:trainSize]

X_test = X_scale[trainSize:]
y_test = y[trainSize:]
# 	# y_train = y_train_val[:val_shape,]
#
# 	# x_val = scaled_train[val_shape:,]
# 	# y_val = y_train_val[val_shape:,]
# x_test = scaled_test
#

# 	#-------------------------------------------------------------------------------
# 	## Create tensorflow models using keras API
	## CNN1
#
# model = Sequential()
# model.add(Conv1D(nodes, kernel_size=3, strides=1,
#              activation='relu',
#              input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=3))
#
# model.add(Dropout(dropRate))
#
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# # model.add(BatchNormalization())
#
# model.add(Dense(128, activation='relu'))
# # model.add(BatchNormalization())
#
# model.add(Dense(num_classes, activation='softmax'))


## DCNN


model = Sequential()
model.add(Conv1D(nodes, kernel_size=3, strides=1,
     activation='relu',
     input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv1D(nodes, kernel_size=3, strides=1,
     activation='relu'))
model.add(Conv1D(nodes, kernel_size=3, strides=1,
         activation='relu'))
model.add(Conv1D(512, kernel_size=1, strides=1,
         activation='relu'))
model.add(MaxPooling1D(pool_size=3))

model.add(Dropout(dropRate))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adam(),
          metrics=['accuracy'])

#-------------------------------------------------------------------------------
## CNN LSTM Architecture

# model = Sequential()
# model.add(Conv1D(nodes, kernel_size=3, strides=1,
# 		 activation='relu',
# 		 input_shape=input_shape))
# # model.add(BatchNormalization())
# model.add(MaxPooling1D(pool_size=2))
#
# # model.add(Dropout(dropRate))
#
# # model.add(Flatten())
# # model.add(Dense(256, activation='relu'))
# # model.add(BatchNormalization())
#
# # model.add(Dense(128, activation='relu'))
# # model.add(BatchNormalization())
# model.add(LSTM(100)) #return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
# # model.add(GlobalMaxPooling1D())
# # model.add(Dense(50, activation='relu'))
# model.add(Dropout(dropRate))
#
# model.add(Dense(num_classes, activation='sigmoid'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#           optimizer=keras.optimizers.Adam(),
#           metrics=['accuracy'])

history = AccuracyHistory()
earlyStop = EarlyStopping(
        monitor='val_acc',
        patience=5,
        mode='max',
		# baseline=0.400,
        verbose=1)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          # verbose=1,
          # validation_data=(x_val, y_val),
          callbacks=[history])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# test_acc.append(score[1])
#




























# print('Test accuracy:', test_acc)
# print('Test accuracy averge:', np.mean(test_acc))
# # plt.plot(range(1,epochs+1), scores, 'b', label="Train")
#
# # plt.plot(range(1,epochs+1), history.acc, 'b', label="Train")
# # plt.plot(range(1,epochs+1), history.val_acc, 'r', label="Test")
# # plt.xlabel('Epochs')
# # plt.ylabel('Accuracy')
# # plt.legend(loc='best')
# # plt.autoscale(enable=True, axis='x', tight=True)
# # plt.title("CNN Learning Rates\nBatch = {0} Epochs = {1}".format(batch_size, epochs) )
# # plt.show()
#
#
# # #-----------------------------------------------------------------------------
# # # Create tensorflow models using API
#
#
# # #   Model 3  (Wide and deep)
# #
#
# #     ## Wide model
# # inputs_1 = tf.keras.Input(shape=(in_size,))
# # x_1 = tf.keras.layers.Dense(nodes, activation=tf.nn.relu)(inputs_1)
# # outputs_1 = tf.keras.layers.Dense(out_size, activation=tf.nn.softmax)(x_1)
# # wide_model = tf.keras.Model(inputs=inputs_1, outputs=outputs_1)
# #
# # wide_model.compile(loss='mse',
# #               optimizer='adam',
# #               metrics=['accuracy'])
# #
# #
# #     ## Deep model
# #
# #
# # deep_inputs = layers.Input(shape=(max_seq_length,))
# # embedding = layers.Embedding(in_size, 8,   input_length=max_seq_length)(deep_inputs)
# # embedding = layers.Flatten()(embedding)
# # embed_out = layers.Dense(1, activation='linear')(embedding)
# # deep_model = Model(inputs=deep_inputs, outputs=embed_out)
# # deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# #
# #  ## Merge
# # merged_out = layers.concatenate([wide_model.output, deep_model.output])
# # merged_out = layers.Dense(1)(merged_out)
# # model_1 = Model(wide_model.input + [deep_model.input], merged_out)
# # model_1.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
# #
# # ## Test and Evel
# # model_1.fit([description_bow_train, variety_train] + [train_embed], labels_train, epochs=10, batch_size=128)
# #
# # # Evaluation
# # model_1.evaluate([description_bow_test, variety_test] + [test_embed], labels_test, batch_size=128)
# #

# # #-----------------------------------------------------------------------------

# #
# # ## This is more complex, explicit tf model.  (Not working yet)
# # # learning_rate = 0.001
# # # epochs = 10
# # # batch_size = 100
# # #
# # # # declare the training data placeholders
# # # # input x - number of distinct features passed into network
# # # x = tf.placeholder(tf.float32, [None, in_size])  # input
# # # # declare the output layer size placeholder
# # # y = tf.placeholder(tf.float32, [None, out_size])  # output
# # #
# # # # Create weights for input to hidden layer.  Stddev is set arbitrarily now
# # # W1 = tf.Variable(tf.random_normal([in_size, nodes], stddev=0.03), name='W1')
# # # b1 = tf.Variable(tf.random_normal([nodes]), name='b1')
# # # #weights connecting the hidden layer to the output layer
# # # W2 = tf.Variable(tf.random_normal([nodes, out_size], stddev=0.03), name='W2')
# # # b2 = tf.Variable(tf.random_normal([out_size]), name='b2')
# # # # calculate the output of the hidden layer
# # # hidden_out = tf.add(tf.matmul(x, W1), b1)
# # # hidden_out = tf.nn.relu(hidden_out)
# # #
# # # output_layer = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
# # #
# # # out_clipped = tf.clip_by_value(output_layer, 1e-10, 0.9999999)
# # # cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(out_clipped)
# # #                          + (1 - y) * tf.log(1 - out_clipped), axis=1))
# # #
# # # optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
# # #
# # # # setup the initialisation operator
# # # init_op = tf.global_variables_initializer()
# # #
# # # # define an accuracy assessment operation
# # # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output_layer, 1))
# # # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# # #
# # # # start training
# # # with tf.Session() as sess:
# # # # initialise the variables
# # #     sess.run(init_op)
# # #     total_batch = int(len(s) / batch_size)
# # #     for epoch in range(epochs):
# # #         avg_cost = 0
# # #         for i in range(total_batch):
# # #             batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
# # #             _, c = sess.run([optimiser, cross_entropy],
# # #                             feed_dict={x: batch_x, y: batch_y})
# # #             avg_cost += c / total_batch
# # #         print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
# # #     print(sess.run(accuracy))
#
# #
# 	# #
# 		# callbacks = [
# 		#     EarlyStopping(
# 		#         monitor='val_acc',
# 		#         patience=10,
# 		#         mode='max',
# 		#         verbose=1),
		#     # ModelCheckpoint(model_path,
		#     #     monitor='val_acc',
		#     #     save_best_only=True,
		#     #     mode='max',
		#     #     verbose=0)
		# ]
