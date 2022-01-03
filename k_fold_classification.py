##############################################################################################################
#
# project/risk_classification/k_fold_classification.py
#
# Here we use a Convolutional Neural Network to do a binary classification. The model will predict whether 
# the dB_H/dt will (1) or will not (0) exceed a threshold level in a future time window. This file, as written
# only examines one threhsold, but this study examines several, and multiple files were written so different models 
# could be trained simultaniously on our server. The model inputs are solar wind paramenters from the OMNI database,
# as well as parameters from the Ottowa (OTT) ground magnetometer station. This model does include the "target"
# parameter dB_H/dt in the training data. In addition, the model was run n times using random weight initalization
# to create uncertainty in the final outputs.
#
##############################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import matplotlib.dates as mdates
import pickle
from pickle import load as pkload
from statistics import mean

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.model_selection import ShuffleSplit

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

# Data directories
projectDir = '~/projects/risk_classification/'

# # stops this program from hogging the GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass



CONFIG = {'thresholds': [9, 18, 42, 66, 90],	# list of thresholds to be examined.
			'params': ['Date_UTC', 'N', 'E', 'sinMLT', 'cosMLT', 'B_Total', 'BY_GSM',
	   					'BZ_GSM', 'flow_speed', 'Vx', 'Vy', 'Vz', 'proton_density', 'Pressure',
	   					'E_Field', 'SYM_H', 'dBHt', 'AE_INDEX', 'SZA'],								# List of parameters that will be used for training. 
	   																							# Date_UTC will be removed, kept here for resons that will be evident below
			'test_storm_stime': ['2011-05-01 00:00:00', '2006-09-01 00:00:00', '2001-01-01 00:00:00'],	# These are the start times for testing storms
			'test_storm_etime': ['2011-11-30 23:59:00', '2007-03-31 23:59:00', '2001-06-30 23:59:00'],	# end times for testing storms. This will remove them from training
			'plot_stime': ['2011-08-05 16:00', '2006-12-14 12:00', '2001-03-30 21:00'],		# start times for the plotting widow. Focuses on the main sequence of the storm
			'plot_etime': ['2011-08-06 18:00', '2006-12-15 20:00', '2001-04-01 02:00'],		# end plotting times
			'plot_titles': ['2011_storm', '2006_storm', '2001_storm'],						# list used for plot titles so I don't have to do it manually
			'forecast': 30,
			'window': 30,																	# time window over which the metrics will be calculated
			'k_fold_splits': 100}															# amount of k fold splits to be performed. Program will create this many models

MODEL_CONFIG = {'time_history': 60, 	# How much time history the model will use, defines the 2nd dimension of the model input array 
					'epochs': 100, 		# Maximum amount of empoch the model will run if not killed by early stopping 
					'layers': 1, 		# How many CNN layers the model will have.
					'filters': 128, 		# Number of filters in the first CNN layer. Will decrease by half for any subsequent layers if "layers">1
					'dropout': 0.2, 		# Dropout rate for the layers
					'initial_learning_rate': 1e-5,		# Learning rate, used as the inital learning rate if a learning rate decay function is used
					'lr_decay_steps':230,}		# If a learning ray decay funtion is used, dictates the number of decay steps



def classification_column(df, param, thresholds, forecast, window):
	'''creating a new column which labels whether there will be a dBT that crosses the threshold in the forecast window.
		Inputs:
		df: the dataframe containing all of the relevent data.
		param: the paramaeter that is being examined for threshold crossings (dBHt for this study).
		thresholds: threshold or list of thresholds to define parameter crossing.
		forecast: how far out ahead we begin looking in minutes for threshold crossings. If forecast=30, will begin looking 30 minutes ahead.
		window: time frame in which we look for a threshold crossing starting at t=forecast. If forecast=30, window=30, we look for threshold crossings from t+30 to t+60
		'''
	
	predicting = forecast+window																# defining the end point of the prediction window
	df['shifted_{0}'.format(param)] = df[param].shift(-forecast)								# creates a new column that is the shifted parameter. Because time moves foreward with increasing
																									# index, the shift time is the negative of the forecast instead of positive.
	indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)						# creates a forward rolling indexer because it won't do it automatically.
	df['window_max'] = df.shifted_dBHt.rolling(indexer, min_periods=1).max()					# creates new coluimn in the df labeling the maximum parameter value in the forecast:forecast+window time frame
	df.reset_index(drop=True, inplace=True)														# just resets the index

	for thresh in thresholds:
		'''This section creates a binary column for each of the thresholds being examined (if multiple). Binary will be 1 if the parameter 
			goes above the given threshold in the time window, and 0 if it does not.'''

		conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions

		binary = [0, 1] 																	# 0 if not cross 1 if cross

		df['{0}_cross'.format(thresh)] = np.select(conditions, binary)						# new column created using the conditions and the binary


	df.drop(['window_max', 'shifted_dBHt'], axis=1, inplace=True)							# removes the two working columns
		
	return df



def omni_prep(path, std_len=10, do_calc=True):

'''Preparing the omnidata for plotting.
	Inputs:
	path: path to project directory
	sdate: storms start date
	edate: storm end date
	std_len: lookback length over which the standard deviation is calculated. i.e. for default
			std_len=30 the standard deviation of a parameter at time t will be calculated from
			df[param][(t-time_history):t].
	do_calc: (bool) is true if the calculations need to be done, false if this is not the first time
			running this specific configuration and a csv has been saved. If this is the case the csv
			file will be loaded.
'''

	if do_calc:

		df = pd.read_csv(path+'../data/omni.csv') 		# loading the omni data

		# reassign the datetime object as the index
		pd.to_datetime(df['Epoch'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Epoch', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index)
		
		df.to_csv(path+'../data/omni_csv_calcs.csv')		# saving so calculations don't have to be done again if program needs to be re-run

	if not do_calc:		# skips the calculations above and loads csv file

		df = pd.read_csv(path+'../data/omni_csv_calcs.csv') # loading the omni data

		# reassign the datetime object as the index
		pd.to_datetime(df['Epoch'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Epoch', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index)

	return df


def data_prep(path, thresholds, params, forecast, window, do_calc=True):
	''' Preparing the magnetometer data for the other functions and concatinating with the other loaded dfs.
		Inputs:
		path: the file path to the project directory
		thresholds(float or list of floats): data threhold beyond which will count as events. 
		params: list of input paramaters to add to the features list. This is done because of 
				memory limitiations and all other columns will be dropped.
		forecast: how far out the data will be examined for a threshold crossing.
		window: the size of the window that will be exaimned for a threshold crossing. i.e. 30 means the maximum 
				value within a 30 minute window will be examined.
		do_calc: (bool) is true if the calculations need to be done, false if this is not the first time
				running this specific configuration and a csv has been saved. If this is the case the csv
				file will be loaded.
	'''
	print('preparing data...')

	if do_calc:
		print('Reading in CSV...')
		df = pd.read_csv(path+'../data/OTT.csv') # loading the station data.
		print('Doing calculations...')
		df['dN'] = df['N'].diff(1) # creates the dN column
		df['dE'] = df['E'].diff(1) # creates the dE column
		df['dBHt'] = np.sqrt(((df['N'].diff(1))**2)+((df['E'].diff(1))**2)) # creates the combined dB/dt column
		df['direction'] = (np.arctan2(df['dN'], df['dE']) * 180 / np.pi)	# calculates the angle of dB/dt
		df['sinMLT'] = np.sin(df.MLT * 2 * np.pi * 15 / 360)
		df['cosMLT'] = np.cos(df.MLT * 2 * np.pi * 15 / 360)

		print('Setting Datetime...')
		# setting datetime index
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)

		print('Getting OMNI data...')
		omnidf = omni_prep(path, do_calc=True)

		print('Concatinating dfs...')
		df = pd.concat([df, omnidf], axis=1, ignore_index=False)	# adding on the omni data

		print('Isolating selected Features...')	# defining the features to be kept
		df = df[params][1:]	# drops all features not in the features list above and drops the first row because of the derivatives

		print('Dropping Nan...')
		datum = df.dropna()		# dropping nan rows
		del df

		print('Creating Classification column...')
		datum = classification_column(datum, 'dBHt', thresholds, forecast=forecast, window=window)		# calling the classification column function
		print('saved CSV')

		datum.to_csv('../data/OTT_prepared.csv') 		# saving CSV for future use

	if not do_calc:		# does not do the above calculations and instead just loads a csv file, then creates the cross column

		datum = pd.read_csv('../data/OTT_prepared.csv')
		datum.reset_index(drop=True, inplace=True)

	return datum


def split_sequences(sequences, result_y1=None, result_y2=None, result_y3=None, result_y4=None, result_y5=None, n_steps=30, include_target=True):
	'''takes input from the data frames and creates the input and target arrays that can go into the models.
		Inputs:
		sequences: dataframe of the input features.
		results_y: series data of the targets for each threshold.
		n_steps: the time history that will define the 2nd demension of the resulting array.
		include_target: true if there will be a target output. False for the testing data.'''

	X, y1, y2, y3, y4, y5 = list(), list(), list(), list(), list(), list()		# creating lists for storing results
	for i in range(len(sequences)-n_steps):										# going to the end of the dataframes
		end_ix = i + n_steps													# find the end of this pattern
		if end_ix > len(sequences):												# check if we are beyond the dataset
			break
		seq_x = sequences[i:end_ix, :]											# grabs the appropriate chunk of the data
		if include_target:
			# gets the appropriate target 
			seq_y1 = result_y1[end_ix]											
			seq_y2 = result_y2[end_ix]
			seq_y3 = result_y3[end_ix]
			seq_y4 = result_y4[end_ix]
			seq_y5 = result_y5[end_ix]
			# appends it to the corresponding list
			y1.append(seq_y1)
			y2.append(seq_y2)
			y3.append(seq_y3)
			y4.append(seq_y4)
			y5.append(seq_y5)
		X.append(seq_x)

	if include_target:
		return np.array(X), np.array(y1), np.array(y2), np.array(y3), np.array(y4), np.array(y5)
	if not include_target:
		return np.array(X)


def prep_test_data(df, stime, etime, thresholds, params, scaler, time_history, prediction_length):
	'''function that segments the selected storms for testing the models. Pulls the data out of the 
		dataframe, splits the sequences, and stores the model input arrays and the real results.
		Inputs: 
		df: Dataframe containing all of the data.
		stime: array of datetime strings that define the start of the testing storms.
		etime: array of datetime strings that define the end of the testing storms.
		thresholds: array on integers that define the crossing binary for each target array. 
		params: list of features to be included as inputs to the models.
		scaler: pre-fit scaler that is uesd to scale teh model input data.
		time_history: amount of time history used to define the 2nd dimension of the model input arrays.
		prediction_length: forecast length+prediction window. Used to cut off the end of the df.'''

	test_dict = {}													# initalizing the dictonary for storing everything
	for start, end, i in zip(stime, etime, range(len(stime))):		# looping through the different storms
		test_dict['storm_{0}'.format(i)] = {}						# creating a sub-dict for this particular storm

		storm_df = df[start:end]									# cutting out the storm from the greater dataframe
		storm_df.reset_index(inplace=True, drop=False)				
		test_dict['storm_{0}'.format(i)]['date'] = storm_df['Date_UTC']		# storing the date series for later plotting
		real_cols = ['Date_UTC', 'dBHt']									# defining real_cols and then adding in the real data to the columns. Used to segment the important data needed for comparison to model outputs
		for thresh in thresholds:
			real_cols.append('{0}_cross'.format(thresh))

		real_df = storm_df[real_cols][time_history:(len(storm_df)-prediction_length)]		# cutting out the relevent columns. trimmed at the edges to keep length consistent with model outputs
		real_df.reset_index(inplace=True, drop=True)

		storm_df = storm_df[params]												# cuts out the model input parameters
		storm_df.drop(storm_df.tail(prediction_length).index,inplace=True)		# chopping off the prediction length. Cannot predict past the avalable data
		storm_df.drop('Date_UTC', axis=1, inplace=True)							# don't want to train on datetime string
		storm_df.reset_index(inplace=True, drop=True)
		storm_df = scaler.transform(storm_df)									# scaling the model input data
		storm_df = split_sequences(storm_df, n_steps=time_history, include_target=False)	# calling the split sequences function to create the additional demension

		test_dict['storm_{0}'.format(i)]['Y'] = storm_df						# creating a dict element for the model input data
		test_dict['storm_{0}'.format(i)]['real_df'] = real_df					# dict element for the real data for comparison

		n_features = test_dict['storm_{0}'.format(i)]['Y'].shape[2]				# grabs the amount of input features for the model to read as input
	
	return test_dict, n_features


def storm_search(data, param, lead, recovery, threshold=-50, patience=5, minimum=20):
	'''Pulling out storms from the training data using a threshold value.
		Inputs:
		data: full dataframe with the testing data removed.
		param: feature to used to identify storms.
		lead: how much time in minutes to add to the beginning of the storms.
		recovery: how much time in minutes to add to the end of the identified storms.
		threhsold: the value the param must get above or below to begin storm identification.
		patience: how long in minutes the param can go below or above the threshold before the storm is defined as ending.
		minimum: minimum length in munutes the storm need to be to be counted. Avoids the issue of interpolation jumps or strage things happening in the data.'''

	storms, y_1, y_2, y_3, y_4, y_5 = list(), list(), list(), list(), list(), list()		# initalizing the lists for storing information.
	for df in data:
		m = 0															# initalizing a counting integer to keep track of the place in the dataframe
		while m < len(df.index):
			if df[param][m] > threshold:								# if the monitored variable passes the threshold value
				n = m													# setting a new counting integer that begins at the current value of m
				while True:												# while the value of param is excedding the threshold
					n = n+1												# increase the counting integer as the loop is continued
					if (n+patience) >= len(df.index):					# breaks the loop if the counting integer excedes the length of the df
						break
					if df[param][n] < threshold:						# allows for temporary dips above the threshold value as long as it does not excede the patience limit
						check = []
						for i in range(0,patience):						# adds elements to a list and then checks to see if the list is longer than the patience limit. Breaks the loop if it is.
							if df[param][n+i] < threshold:
								check.append(df[param][n+i])
						if len(check) == patience:
							break
						else:
							continue
				possible_storm=df[m-lead:n+recovery]					# adding the lead and recovery time to the extracted storm
				length = lead+minimum+recovery							# identifies total length of the segmented storm
				if possible_storm.shape[0] > length:					# checks to make sure it meets the minimum length
					possible_storm.reset_index(inplace = True, drop = True)
					possible_storm.drop(['Date_UTC'], axis=1, inplace=True)
					y_1.append(to_categorical(possible_storm['9_cross'].to_numpy(), num_classes=2))			# turns the one demensional resulting array for the storm into a
					y_2.append(to_categorical(possible_storm['18_cross'].to_numpy(), num_classes=2))		# catagorical array for training
					y_3.append(to_categorical(possible_storm['42_cross'].to_numpy(), num_classes=2))
					y_4.append(to_categorical(possible_storm['66_cross'].to_numpy(), num_classes=2))
					y_5.append(to_categorical(possible_storm['90_cross'].to_numpy(), num_classes=2))
					possible_storm.drop(['9_cross', '18_cross', '42_cross', '66_cross', '90_cross'], axis=1, inplace=True)
					storms.append(possible_storm)
				m=n+recovery										# resetting the counting integer 
			else:
				m+=1

	return storms, y_1, y_2, y_3, y_4, y_5


def prep_train_data(df, stime, etime, time_history):
	''' function that prepares the training data.
		Inputs:
		df: the full, prepared dataframe.
		time_history: amount of time history to be included as input to the model.
		lead: how much time in hours to add to the beginning of the storm.
		recovery: how much recovery time in hours to add to the end of the storm.
	'''

	# using date time index so we can segment out the testing data
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	# Very lazy way to do this but these are the segments that are not those segmented for testing
	start = sorted([((datetime.strptime(s, '%Y-%m-%d %H:%M:%S'))) for s in stime])		# larger df is in time order, the sorted list is used to avoid removing incorrect data
	end = sorted([((datetime.strptime(e, '%Y-%m-%d %H:%M:%S'))) for e in etime])
	data = []								# creats a list to store the seperate dfs that will be segmented for training. Must be kept seperate due to importance of time series 
	for i in range(len(start)):
		if i == 0:
			data.append(df[df.index < start[i]])
		if (i > 0) & (i < len(start)-1):
			data.append(df[(df.index > end[i-1]) & (df.index < start[i])])
		elif i == len(start)-1: 
			data.append(df[(df.index > end[i-1]) & (df.index < start[i])])
			data.append(df[df.index > end[i]])

	# resetting the indexes. The sequence_splitting and storm_search functions are not written to handle datetime index
	for df in data:
		df.reset_index(inplace=True, drop=False)

	print('\nFinding storms...')
	storms, y_1, y_2, y_3, y_4, y_5 = storm_search(data, 'AE_INDEX', 60, 0, threshold=1000, patience=720, minimum=120)			# calling the storms search function
	print('Number of storms: '+str(len(storms)))

	to_scale_with = pd.concat(storms, axis=0, ignore_index=True)					# finding the largest storm with which we can scale the data. Not sure this is the best way to do this
	scaler = StandardScaler()									# defining the type of scaler to use
	print('Fitting scaler')
	scaler.fit(to_scale_with)									# fitting the scaler to the longest storm
	print('Scaling storms train')
	storms = [scaler.transform(storm) for storm in storms]		# doing a scaler transform to each storm individually

	train_dict = {}												# creatinga  training dictonary for storing everything
	Train, train1, train2, train3, train4, train5 = np.empty((1,60,18)), np.empty((1,2)), np.empty((1,2)), np.empty((1,2)), np.empty((1,2)), np.empty((1,2))	# creating empty arrays for storing sequences
	for storm, y1, y2, y3, y4, y5, i in zip(storms, y_1, y_2, y_3, y_4, y_5, range(len(storms))):		# looping through the storms
		X, x1, x2, x3, x4, x5 = split_sequences(storm, y1, y2, y3, y4, y5, time_history)				# splitting the sequences for each storm individually

		# concatiningting all of the results together into one array for training
		Train = np.concatenate([Train, X])
		train1 = np.concatenate([train1, x1])
		train2 = np.concatenate([train2, x2])
		train3 = np.concatenate([train3, x3])
		train4 = np.concatenate([train4, x4])
		train5 = np.concatenate([train5, x5])

	# adding all of the training arrays to the dict
	train_dict['X'] = Train
	train_dict['9_thresh'] = train1
	train_dict['18_thresh'] = train2
	train_dict['42_thresh'] = train3
	train_dict['66_thresh'] = train4
	train_dict['90_thresh'] = train5
	n_features = train_dict['X'].shape[2]				# grabbing the number of input columns in the training data for input into the model


	print('Finished calculating percent')

	return train_dict, scaler, n_features


def create_CNN_model(model_config, n_features, loss='mse', early_stop_patience=3):
	'''Initializing our model
		Inputs:
		model_config: predefined model configuration dictonary
		n_features: amount of input features into the model
		loss: loss function to be uesd for training
		early_stop_patience: amount of epochs the model will continue training once there is no longer val loss improvements
		'''


	model = Sequential()						# initalizing the model

	model.add(Conv2D(model_config['filters'], (1,2), padding='same',
									activation='relu', input_shape=(model_config['time_history'], n_features, 1)))			# adding the CNN layer
	model.add(MaxPooling2D())						# maxpooling layer reduces the demensions of the training data. Speeds up models and improves results
	model.add(Flatten())	
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))				# softmax activation for binary classification
	opt = tf.keras.optimizers.Adam(learning_rate=1e-6)		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ['accuracy'])					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting


	return model, early_stop


def fit_CNN(model, xtrain, xval, ytrain, yval, model_config, early_stop, thresh, split, first_time=True):
	'''Performs the actual fitting of the model.
	Inputs:
	model: model as defined in the create_model function.
	xtrain: training data inputs
	xval: validation inputs
	ytrain: training target vectors
	yval: validation target vectors
	model_config: dictonary of model parameters
	early_stop: predefined early stopping function
	thresh: threshold being trained on. Used for saving models
	split: split being trained. Used for training.
	first_time: if True model will be trainined, False model will be loaded.'''

	if first_time:

		# reshaping the model input vectors for a single channel
		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))

		model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
					verbose=1, shuffle=True, epochs=model_config['epochs'], callbacks=[early_stop])			# doing the training! Yay!

		# calibrator = CalibratedClassifierCV(model, cv='prefit')
		# calibrator.fit(xval, yval)

		model.save('models/{0}_model_CNN_split_{1}.h5'.format(thresh, split))		# saving the model


	if not first_time:

		model = load_model('models/{0}_model_CNN_split_{1}.h5'.format(thresh, split))						# loading the models if already trained

	return model


def making_predictions(model, test_dict, threshold, split):
	'''function used to make the predictions with the testing data
		Inputs:
		model: pre-trained model
		test_dict: dictonary with the testing model inputs and the real data for comparison
		thresholds: which threshold is being examined
		split: which split is being tested'''

	for key in test_dict:									# looping through the sub dictonaries for each storm

		Xtest = test_dict[key]['Y']							# defining the testing inputs
		Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1))				# reshpaing for one channel input
	
		predicted = model.predict(Xtest, verbose=1)						# predicting on the testing input data

		predicted = tf.gather(predicted, [1], axis=1)					# grabbing the positive node
		predicted = predicted.numpy()									# turning to a numpy array
		predicted = pd.Series(predicted.reshape(len(predicted),))		# and then into a pd.series

		df = test_dict[key]['real_df']									# calling the correct dataframe
		df['{0}_predicted_split_{1}'.format(threshold, split)] = predicted		# and storing the results 

	return test_dict



def main(path, config, model_config, first_time=True):
	'''Here we go baby! bringing it all together.
		Inputs:
		path: path to the data.
		config: dictonary containing different specifications for the data prep and other things that aren;t the models themselves.
		model_config: dictonary containing model specifications.
		first_time: if True the model will be training and the data prep perfromed. If False will skip these stpes and I probably messed up the plotting somehow.
		'''

	print('Entering main...')
	print('First time: '+str(first_time))
	
	splits = config['k_fold_splits']		# denines the number of splits
	if first_time==True:					# If this is the first time we're going through this.
		df = data_prep(path, config['thresholds'], config['params'], config['forecast'], config['window'], do_calc=True)		# calling the data prep function
		train_dict, scaler, n_features = prep_train_data(df, config['test_storm_stime'], config['test_storm_etime'], 
												model_config['time_history'])  												# calling the training data prep function
		with open('models/standardscaler.pkl', 'wb') as f:					# saving the scaler in case I have to run this again
			pickle.dump(scaler, f)	
	if first_time==False:													# loads the scaler if this is not the first time we have run this
		with open('models/standardscaler.pkl', 'rb') as f:
			scaler = pickle.load(f)
	
	if first_time==True:				# Goes through all the model training processes if first time going through model

		train_indicies = pd.DataFrame()
		val_indicies = pd.DataFrame()

		test_dict, n_features = prep_test_data(df, config['test_storm_stime'], config['test_storm_etime'], config['thresholds'], config['params'], 
									scaler, model_config['time_history'], prediction_length=config['forecast']+config['window'])						# processing the tesing data

		sss = ShuffleSplit(n_splits=splits, test_size=0.2, random_state=12)								# defines the lists of training and validation indicies to perform the k fold splitting
		X = train_dict['X']																# grabbing the training data for model input

		y = train_dict['{0}_thresh'.format(thresh)]										# grabbing the target arrays for training
		train_index, val_index = [], []							# initalizes lists for the indexes to be stored
		for train_i, val_i in sss.split(y):						# looping through the lists, adding them to other differentiated lists
			train_index.append(train_i)
			val_index.append(val_i)

		for train_i, val_i, split in zip(train_index, val_index, range(splits)):		# saving this traiing and validation indicies for each split for possible anaylsis
			train_indicies['split_{0}'.format(split)] = train_i
			val_indicies['split_{0}'.format(split)] = val_i

		train_indicies.to_csv('outputs/train_indicies.csv')
		val_indicies.to_csv('outputs/val_indicies.csv')


		for train_i, val_i, split in zip(train_index, val_index, range(splits)):		# this is the bulk of the K fold. We loop through the list of indexes and train on the different train-val indices

			tf.keras.backend.clear_session() 				# clearing the information from any old models so we can run clean new ones.
			MODEL, early_stop = create_CNN_model(model_config, n_features, loss='mse', early_stop_patience=10)					# creating the model

			# pulling the data and catagorizing it into the train-val pairs
			xtrain = X[train_i]
			xval =  X[val_i]
			ytrain = y[train_i]
			yval = y[val_i]

			model = fit_CNN(MODEL, xtrain, xval, ytrain, yval, model_config, early_stop, thresh, split, first_time=True)			# does the model fit!

			test_dict = making_predictions(model, test_dict, thresh, split)					# defines the test dictonary for storing results

		for i in range(len(test_dict)):
			test_dict['storm_{0}'.format(i)]['real_df'].to_csv('outputs/storm_{0}_k_fold_results.csv'.format(i))		# saving the real and model predicted results for each split and storm




if __name__ == '__main__':

	main(projectDir, CONFIG, MODEL_CONFIG, first_time=True)		# calling the main function.

	print('It ran. Good job!')














