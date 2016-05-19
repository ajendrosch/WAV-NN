from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU

import numpy as np
# import keras necessary classes
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils

def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
	for cur_unit in xrange(num_recurrent_units):
		model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model

def create_gru_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
	for cur_unit in xrange(num_recurrent_units):
		model.add(GRU(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model

def create_cnn_network():
	# Creating the model which consists of 3 conv layers followed by
	# 2 fully conntected layers
	print('creating the model')

	# Sequential wrapper model
	model = Sequential()

	# first convolutional layer
				#out,in,
	model.add(Convolution1D(32,10,input_shape=(10, 32)))
	model.add(Activation('relu'))

	# second convolutional layer
	model.add(Convolution1D(48, 32))
	model.add(Activation('relu')) 
	model.add(MaxPooling1D())

	# third convolutional layer
	model.add(Convolution1D(32, 48))
	model.add(Activation('relu'))
	model.add(MaxPooling1D())

	# convert convolutional filters to flatt so they can be feed to 
	# fully connected layers
	model.add(Flatten())

	# first fully connected layer
	model.add(Dense(32*6*6, 128, init='lecun_uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	# second fully connected layer
	model.add(Dense(128, 128, init='lecun_uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	# last fully connected layer which output classes
	model.add(Dense(128, 10, init='lecun_uniform'))
	model.add(Activation('softmax'))

	# setting sgd optimizer parameters
	sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd)
	return model
