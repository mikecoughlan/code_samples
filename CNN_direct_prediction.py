
##############################################################################################################
#
# project/GEM_challenge_paper/CNN_ver_3.py
#
# Script that uses an Convolutional Neural Network to predict dB_H/dt using solar wind and limited magnetometer data.
# This version will take the abs(Vx) to avoid the largest Vx value being 0. dB/dt is predicted one minute ahead 
# without using the target parameter as an input. Has 6 testing storms as identified by the GEM challenge in
# Pulkkinen et al.(2013), with OMNI data used for 5 of the storms, and ACE data for the October 2003 storm, 
# as OMNI has very little data for that storm. Flat 15 minute propgation was used on the ACE data to attempt
# to match the propogation performed on the OMNI data from L1 to the Bow Shock.
# 
# Version adding in the halloween storm to the testing pool and removing from training
#
##############################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_recall_curve, f1_score, auc

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
import tensorflow as tf

# Data directories
projectDir = '~/projects/GEM_chal_paper/'

# # stops this program from using more of the GPU than it needs.
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

CONFIG = {'thresholds': [18, 42, 66, 90],	# list of thresholds to be examined.
			'test_storm_stime': ['2006-12-14 12:00:00', '2001-08-31 00:00:00', '2005-08-31 10:00:00',
									'2010-04-05 00:00:00','2011-08-05 09:00:00', '2003-10-29 06:00:00'],	# These are the start times for testing storms
			'test_storm_etime': ['2006-12-16 00:00:00', '2001-09-01 00:00:00', '2005-09-01 12:00:00', 
									'2010-04-06 00:00:00', '2011-08-06 09:00:00', '2003-10-30 06:00:00'],	# end times for testing storms. This will remove them from training
			'plot_titles': ['December 2006', 'August 2001', 'August 2005', 'April 2010', 'August 2011', 'October 2003'],		# list used for plot titles so I don't have to do it manually
			'window': 20,																					# time window over which the metrics will be calculated
			'lead':12,																				# lead: how much time in hours to add to the beginning of the storm.
			'recovery':24, 
			'disc': 'new_2003_storm'}																			# recovery: how much recovery time in hours to add to the end of the storm.												

MODEL_CONFIG = {'time_history': 60, 	# How much time history the model will use, defines the 2nd dimension of the model input array 
					'epochs': 200, 		# Maximum amount of empoch the model will run if not killed by early stopping 
					'layers': 1, 		# How many CNN layers the model will have.
					'filters': 512, 		# Number of filters in the first CNN layer. Will decrease by half for any subsequent layers if "layers">1
					'dropout': 0, 		# Dropout rate for the LSTM layers
					'initial_learning_rate': 1e-6,		# Learning rate, used as the inital learning rate if a learning rate decay function is used
					'lr_decay_steps':230,}		# If a learning ray decay funtion is used, dictates the number of decay steps



def omni_prep(path):

	'''Preparing the omnidata for concatination. This is where any calculations on the data (std, d/dt, etc.) can be performed.
		Inputs:
		path: path to project directory'''

	df = pd.read_feather('../data/omniData.feather') # loading the omni data

	df.reset_index(drop=True, inplace=True) # reseting the index so its easier to work with integer indexes

	df['Vx'] = df['Vx'].abs()			# taking the absolute value of the Vx var

	# reassign the datetime object as the index. Need to do this to properly concatinate the OMNI and Supermag dataframes
	pd.to_datetime(df['Epoch'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('Epoch', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	cols = [col for col in df.columns]			# creates a list to store the column names from the omni df for later use. 

	return df, cols


def ace_prep(path):

	'''Preparing the ace data for testing on the 2003 halloween storm.
		Inputs:
		path: path to project directory
	'''

	df = pd.read_csv('../data/ace-omni-2003-storm.csv') # loading the omni data

	df.reset_index(drop=True, inplace=True) # reseting the index so its easier to work with integer indexes

	df['Vx'] = df['Vx'].abs()			# taking the absolute value of the Vx var

	# reassign the datetime object as the index. Need to do this to properly concatinate the OMNI and Supermag dataframes
	pd.to_datetime(df['ACEepoch'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('ACEepoch', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	cols = [col for col in df.columns]				# creates a list to store the column names from the omni df for later use. 

	return df, cols


def data_prep(path, thresholds, station, do_calc=True):
	''' Preparing the magnetometer data for the other functions and concatinating with the other loaded dfs.
		Inputs:
		path: the file path to the project directory
		thresholds(float or list of floats): data threhold beyond which will count as events. 
		params: list of input paramaters to add to the features list. This is done because of 
				memory limitiations and all other columns will be dropped.
		station: the ground magnetometer station being examined. 
		do_calc: (bool) is true if the calculations need to be done, false if this is not the first time
				running this specific configuration and a csv has been saved. If this is the case the csv
				file will be loaded.
	'''
	print('preparing data...')

	if do_calc:
		print('Reading in feather...')
		df = pd.read_feather('../data/{0}.feather'.format(station)) 	# loading the station data.
		print('Doing calculations...')
		df['dBHt'] = np.sqrt(((df['dbn_nez'].diff(1))**2)+((df['dbe_nez'].diff(1))**2)) 	# creates the combined dB/dt column
		df.drop(['dbn_nez','dbe_nez', 'dbz_nez'], axis=1, inplace=True)

		print('Setting Datetime...')

		# setting datetime index
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)

		print('Getting OMNI data...')

		# calling the omni prep function
		omnidf, cols = omni_prep(path)
		acedf, cols = ace_prep(path)

		print('Concatinating dfs...')

		# concatinating the df's together
		omni_df = pd.concat([df, omnidf], axis=1, ignore_index=False)
		ace_df = pd.concat([df, acedf], axis=1, ignore_index=False)

		columns = cols + ['SZA', 'cosMLT', 'sinMLT']


		print('Isolating selected Features...')	
		# defining the features to be kept
		omni_df = omni_df[1:]	# drops all features not in the features list above and drops the first row because of the derivatives
		ace_df = ace_df[1:]	# drops all features not in the features list above and drops the first row because of the derivatives

		print('Dropping Nan...')
		# dropping nan rows and columns not selected for input
		datum = omni_df.dropna(subset=columns)
		ace_datum = ace_df.dropna(subset=columns)
		del omni_df 		# try to save some memory
		del ace_df

		datum.to_csv('../data/{0}_prepared.csv'.format(station))
		ace_datum.to_csv('../data/{0}_prepared_ace.csv'.format(station))

	if not do_calc:		# does not do the above calculations and instead just loads a csv file, then creates the cross column

		datum = pd.read_csv('../data/{0}_prepared.csv'.format(station))		# loading omni df
		datum.drop('Unnamed: 0', inplace=True, axis=1)			# dropping the column that shows up after importing the csv file 
		datum.reset_index(drop=True, inplace=True)					

		ace_datum = pd.read_csv('../data/{0}_prepared_ace.csv'.format(station))		# loading ace df
		ace_datum.drop('Unnamed: 0', inplace=True, axis=1)
		ace_datum.reset_index(drop=True, inplace=True)


	return datum, ace_datum


def split_sequences(sequences, results=None, n_steps=30, include_target=True):
	''' Transforms the dataframe into a numpy array of shape (n, time_history, n_features).
		Inputs:
		sequences: dataframe containing input features
		results: series of target variables (dB/dt) 
		n_steps: time history that will be used to extend the array to 3 dimensions
		include_targets: if True, will include the target output. False for testing data.
	'''
	X, y = list(), list()
	for i in range(len(sequences)-n_steps):
		end_ix = i + n_steps	# find the end of this pattern
		end_iy = i + n_steps+1	# find the end of this pattern
		if end_iy >= len(sequences):		# check if we are beyond the dataset
			break
		seq_x = sequences[i:end_ix, :]		# gather input and output parts of the pattern
		
		if include_target:
			seq_y = results[end_iy]		# gets the target variable for the end of the sequence
			if not np.isnan(seq_y):		# checks if there is missing data so we don't train on it
				X.append(seq_x)			# creates 2d array of shape (n_steps, n_features)
				y.append(seq_y)			# adds the traget array if not nan

		if not include_target:		# doesn't create the target array 
			X.append(seq_x)

	if include_target:
		return np.array(X), np.array(y)

	if not include_target:
		return np.array(X)


def prep_test_data(df, ace_df, stime, etime, time_history, scaler, station):
	''' Prepares the testing data to be put into the predict function.
		Inputs:
		df: the combined set of data from which the testing data will be extracted
		stime: start time for the storms to be extracted
		etime: end time for storms 
		params: list of input features to be included
		time_history: amount of time history to be used by the models
		scaler: scaler used to scale the data.
		station: magnetometer station being analyzed.
	'''
	# pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	# df.reset_index(drop=True, inplace=True)
	# df.set_index('Date_UTC', inplace=True, drop=True)
	# df.index = pd.to_datetime(df.index)

	test_dict = {}		# establishing a new dictonary to store all the results
	for start, end, i in zip(stime, etime, range(len(stime))):					# using i here puts this out of wack with the GEM challenge numbers. To convert, add 2 to the storm number
		if (start == '2011-08-05 09:00:00') & (station == 'PBQ'):					# PBQ station did not take data for this storm
			continue
		test_dict['storm_{0}'.format(i)] = {} 	# creating dict for each storm
		if (start == '2003-10-29 06:00:00'):		# grabs the ACE data for this storm. There is very little OMNI data for this storm
			storm_df = ace_df[start:end]
		else:
			storm_df = df[start:end]		# pulling out the storm data and hten reseting the index 
		storm_df.reset_index(inplace=True, drop=False)
		test_dict['storm_{0}'.format(i)]['date'] = storm_df['Date_UTC']		# saving the date column for later plotting
		real_cols = ['Date_UTC', 'dBHt']	# coulmns to be saved as the real results for later comparison

		real_df = storm_df[real_cols][(time_history+1):len(storm_df)]		# have to trim the target column because the sequence splitting will cut off the first n_steps 
		real_df.reset_index(inplace=True, drop=True)

		print('Storm df columns')
		storm_df.drop(['Date_UTC', 'dBHt'], axis=1, inplace=True)				# dropping extra columns
		if 'index' in storm_df.columns:																	# removing the 'index' column if it somehow got through
			storm_df.drop(['index'], axis=1, inplace=True)
		storm_df.reset_index(inplace=True, drop=True)
		storm_df = scaler.transform(storm_df)		# scaling the data using the scaler fit on the training data
		storm_df = split_sequences(storm_df, n_steps=time_history, include_target=False)		# calling the split sequences function without producing a target array

		test_dict['storm_{0}'.format(i)]['Y'] = storm_df	# storing testing input data
		test_dict['storm_{0}'.format(i)]['real_df'] = real_df		# storing the real data for this storm

		n_features = test_dict['storm_{0}'.format(i)]['Y'].shape[2]		# defing the number of features included. Needed for model defenition
	
	return test_dict, n_features


def storm_search(data, param, lead, recovery, threshold=-50, patience=5, minimum=20):
	'''Pulling out storms using a threhsold vlue and creating a training set.
		Inputs:
		data: dataframe of OMNI and Supermag data with teh test set's already removed. 
		param: paramater to use for storm flagging (SYM-H, AE_INDEX, etc.)
		lead: how much time in minutes to add to the beginning of the storm.
		recovery: how much recovery time to add to the end of the storm.
		threshold: What the flag treshold should be for the given parameter to identify storms.
		patience: How much time the storm can go above or below the threshold value before the storm is cutoff.
		minimum: storm must be of length greater than this value to be added to the training set. 
		'''
	storms, y = list(), list()
	for df in data: 	# testing data being removed leaves more than one dataframe
		m = 0 		# counting variable used for indexing
		while m < len(df.index):	# setting the stopping point
			if df[param][m] > threshold:	# if the monitored variable passes the threshold value
				n = m 			# creating a new variable for counting in this loop
				while True:		# while the parameter is above or below the threshold
					n = n+1
					if (n+patience) >= len(df.index):  		# setting a cut off point so that we don't index past the length of the dataframe
						break
					if df[param][n] < threshold:  			# allows for temporary dips above the threshold value as long as it does not excede the patience limit
						check = []
						for i in range(0,patience):			# this whoe part adds values to a list and then checks to see if the list is longer than the patience limit
							if df[param][n+i] < threshold:	# if it is longer, it breakes the loop, if shorter it continues
								check.append(df[param][n+i])
						if len(check) == patience:				# exits the loop if the param is below the threshold for more than the patience limit
							break
						else:
							continue
				possible_storm=df[m-lead:n+recovery]		# adding the lead and recovery time to the extracted storm
				length = lead+minimum+recovery
				if possible_storm.shape[0] > length:		# checks to see if the storm meets the minimum time requirements
					possible_storm.reset_index(inplace = True, drop = True)
					y.append(possible_storm['dBHt'].to_numpy())		# adds the target column to the y list
					possible_storm.drop(['dBHt', 'Date_UTC', 'AE_INDEX'], axis=1, inplace=True)  	# removing the target variable from the storm data so we don't train on it
					storms.append(possible_storm)		# adds the storm to the list of storms
				m=n+recovery  		# resets the the index variable so the storm doesn't get double counted
			else:
				m+=1  		# moves the index along if there is no threshold exceedence 

	return storms, y


def storm_extract(data, storm_list, lead, recovery):
	'''Pulling out storms using a defined list of datetime strings, adding a lead and recovery time to it and 
		appending each storm to a list which will be later processed.
		Inputs:
		data: dataframe of OMNI and Supermag data with teh test set's already removed. 
		storm_list: datetime list of storms minimums as strings.
		lead: how much time in hours to add to the beginning of the storm.
		recovery: how much recovery time in hours to add to the end of the storm.
		'''
	storms, y = list(), list()					# initalizing the lists
	df = pd.concat(data, ignore_index=True)		# putting all of the dataframes together, makes the searching for stomrs easier

	# setting the datetime index
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	stime, etime = [], []						# will store the resulting time stamps here then append them to the storm time df
	for date in storm_list:					# will loop through the storm dates, create a datetime object for the lead and recovery time stamps and append those to different lists
		stime.append((datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))-pd.Timedelta(hours=lead))
		etime.append((datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))+pd.Timedelta(hours=recovery))
	# adds the time stamp lists to the storm_list dataframes
	storm_list['stime'] = stime
	storm_list['etime'] = etime
	for start, end in zip(storm_list['stime'], storm_list['etime']):			# looping through the storms to remove the data from the larger df
		storm = df[(df.index >= start) & (df.index <= end)]
		print('Storm length: '+str(len(storm))+' : '+str(start))
		if len(storm) != 0:
			storms.append(storm)			# creates a list of smaller storm time dataframes
	for storm in storms:
		storm.reset_index(drop=True, inplace=True)		# resetting the storm index and simultaniously dropping the date so it doesn't get trained on
		y.append(storm['dBHt'].to_numpy())				# creating the traget array
		storm.drop(['dBHt'], axis=1, inplace=True)  	# removing the target variable from the storm data so we don't train on it

	return storms, y



def prep_train_data(df, stime, etime, lead, recovery, time_history):
	''' function that prepares the training data.
		Inputs:
		df: the full, prepared dataframe.
		time_history: amount of time history to be included as input to the model.
		lead: how much time in hours to add to the beginning of the storm.
		recovery: how much recovery time in hours to add to the end of the storm.
	'''

	# setting the index as datetime so that the testing storms can be pulled from the full dataset
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	# Removing data not segmented for testing
	start = sorted([((datetime.strptime(s, '%Y-%m-%d %H:%M:%S'))) for s in stime])			# larger df is in time order, the sorted list is used to avoid removing incorrect data
	end = sorted([((datetime.strptime(e, '%Y-%m-%d %H:%M:%S'))) for e in etime])
	data = []											# creats a list to store the seperate dfs that will be segmented for training. Must be kept seperate due to importance of time series 
	for i in range(len(start)):
		if i == 0:
			data.append(df[df.index < start[i]])
		if (i > 0) & (i < len(start)-1):
			data.append(df[(df.index > end[i-1]) & (df.index < start[i])])
		elif i == len(start)-1: 
			data.append(df[(df.index > end[i-1]) & (df.index < start[i])])
			data.append(df[df.index > end[i]])


	# # resetting the indexes. The sequence_splitting and storm_search functions are not written to handle datetime index
	for df in data:
		df.reset_index(inplace=True, drop=False)


	print('Loading storm list...')
	storm_list = pd.read_csv('stormList.csv', header=None, names=['dates'])		# loading the list of storms as defined by SYM-H minimum
	for i in range(len(storm_list)):						# cross checking it with testing storms, dropping storms if they're in the test storm list
		d = datetime.strptime(storm_list['dates'][i], '%Y-%m-%d %H:%M:%S')		# converting list of dates to datetime
		for s, e, in zip(start, end):									# drops any storms in the list that overlap with the testing storms
			if (d >= s) & (d <= e):
				storm_list.drop(i, inplace=True)
				print('found one! Get outta here!')

	dates = storm_list['dates']				# just saving it to a variable so I can work with it a bit easier

	print('\nFinding storms...')
	# storms, y = storm_search(data, 'AE_INDEX', 60, 0, threshold=1000, patience=720, minimum=120)		# extracting the storms using search method
	storms, y = storm_extract(data, dates, lead=lead, recovery=recovery)		# extracting the storms using list method
	print('Number of storms: '+str(len(storms)))		# just printing out how many storms were extracted

	to_scale_with = pd.concat(storms, axis=0, ignore_index=True)		# creating one large df to scale with
	scaler = MinMaxScaler() 					# found that this is one of the best scalers for the CNN with this much data
	print('Fitting scaler...')
	scaler.fit(to_scale_with)					# doing the scaler fit
	print('Scaling storms training data...')
	storms = [scaler.transform(storm) for storm in storms]		# fitting to all of the storms individually
	n_features = storms[1].shape[1]				# identifying how many features (columns) are in being used.

	train_dict = {}			# establishing a dict for the training data
	train_X, train_y = np.empty((1, time_history, n_features)), np.empty((1))		# creating empty arrays to combine all storm data
	for storm, storm_y, i in zip(storms, y, range(len(storms))):
		X, x1 = split_sequences(storm, storm_y, time_history)		# calling the split_sequences function for each of the storms. Doing them individually prevents data leakage
		if len(X) != 0:								# checking to make sure not everything got dropped in the split sequence function
			train_X = np.concatenate([train_X, X])						# concatinating all of the storm data together now that they've been put into 3D arrays
			train_y = np.concatenate([train_y, x1])


	train_dict['X'] = train_X						# adding training data to training dict
	train_dict['y'] = train_y						# adding target variable to training dict

	return train_dict, scaler


def create_CNN_model(model_config, n_features, loss='mse', early_stop_patience=3):
	'''Initializing our model
		Inputs: 
		model_config: list of model configuration variables to be used in designing the model
		n_features: the amount of input features that will be used in the model
		loss: the loss function the model will use. defaul is mean squared error
		early_stop_patience: this one seems self explanitory
		'''

	model = Sequential()

	model.add(Conv2D(model_config['filters'], (1,2), padding='same',
			activation='relu', input_shape=(model_config['time_history'], n_features, 1)))		# fitting the model. The (1,2) is the sliding window that 
														# moves over the data. 
	model.add(MaxPooling2D())										# MaxPool layer reduces the data and makes the model run faster
	model.add(Flatten())											# reducing the demensionallity of the data, shows imporvement over models without this layer
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='relu'))								# final layer, can play with the activation a bit here


	opt = tf.keras.optimizers.Adam(learning_rate=1e-6)								# learning rat that showed improvement over the CNN
	model.compile(optimizer=opt, loss=loss)					
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# stops the model from overfitting if it doesn't improve the loss after some number of epochs 

	return model, early_stop


def fit_CNN(model, train_dict, model_config, early_stop, station, first_time=True, val_size=0.3):
	'''Here we go! Function that fits the model to the training data, using validation data to prevent overfitting.
		Inputs:
		model: the model created in the create_model function
		train_dict: dictonary contining all of the training information
		model_config: dict with model specific information
		early_stop: the early stop function that also helps prevent overfitting
		first_time: True is the model is being loaded, False if needs to be trainied
		val_size: share of the data that will be used for validation
		'''

	print('len train_dict: '+str(len(train_dict)))

	if first_time:

		xtrain, xval, ytrain, yval = train_test_split(train_dict['X'], train_dict['y'], test_size=val_size, shuffle=True)

		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))		# reshpaing the input training data. The 1 indicates how many "color" filters (that's not the right word)
		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))				# reshaping the input validation data

		model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
					verbose=1, shuffle=True, epochs=model_config['epochs'], callbacks=[early_stop])			# fitting the model using early stopping and validation data

		model.save('models/CNN_{0}_SYM_updated.h5'.format(str(station)))	
		history = model.history							# getting the model history(loss)

		plt.plot(history.history['loss'])						# plotting the val loss vs loss
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper right')
		plt.savefig('plots/training_loss_{0}_SYM_updated.png'.format(station))			# saving the model loss figure
												

	if not first_time:

		model = load_model('models/CNN_{0}_SYM_updated.h5'.format(station))			# loading the already fitted model

	return model


def making_predictions(model, test_dict):
	'''using the fitted models to mke predicitons on the testing data!
		Inputs:
		model: the fitted model
		test_dict: dict for the testing data
		'''

	for key in test_dict:

		Xtest = test_dict[key]['Y']								# loading the correct testing data from the testing dict
		Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1))		# reshaping properly
	
		predicted = model.predict(Xtest, verbose=1)						# predicting

		predicted = tf.gather(predicted, [0], axis=1)					# model outputs a tensorflow tensor, this grabs the output data	
		predicted = predicted.numpy()							# turns it into a numpy array
		predicted = pd.Series(predicted.reshape(len(predicted),)) 			# then turns it into a pd.series

		df = test_dict[key]['real_df']							# re-orienting ourselves in the sub dict containing the real data

		df['predicted'] = predicted	 						# storing the predictions in the same dataframe


	return test_dict



def calculating_classification_scores(data, thresholds, window):
	'''Doing the calcs for differnet metric scores. Does not include the AUC score
		Inputs:
		df: the results dataframes
		thresholds: list of thresholds to be tested against
		window: window in which the max values will be calculated
		'''
	pod, pofd, fb, hss, rmse = [], [], [], [], []
	pred_max, real_max = [], []

	data.dropna(inplace=True)
	pr = data['predicted']  							# grabbing the predicted thresholds
	re = data['dBHt']  									# grabbing the real data 

	rmse.append(np.sqrt(mean_squared_error(re,pr)))  	# straight calculation of the root mean squared error

	for i in range(0, len(data), window):				# written for non-overlapping windows. For continuious moving windows, remove the window from the range
		pred = np.abs(pr[i:i+window]).max()			# grabbing the max predicted value in the window
		real = np.abs(re[i:i+window]).max()			# grabbing the max real value in the window
		pred_max.append(pred)						# appending the results to a list
		real_max.append(real)

	df = pd.DataFrame({'real_max':real_max, 
						'pred_max':pred_max})  		# putting these in a df because it makes the next step faster


	for thresh in thresholds:

		# seperating the dataframe into the confusion matrix values
		A = df[(df['pred_max'] >= thresh) & (df['real_max'] >= thresh)]
		B = df[(df['pred_max'] >= thresh) & (df['real_max'] < thresh)]
		C = df[(df['pred_max'] < thresh) & (df['real_max'] >= thresh)]
		D = df[(df['pred_max'] < thresh) & (df['real_max'] < thresh)]
		
		a, b, c, d = len(A), len(B), len(C), len(D)			# getting the lengths and using the length as the matrix value
		print('threshold: '+str(thresh))
		print("len a: "+str(len(A)))
		print("len b: "+str(len(B)))
		print("len c: "+str(len(C)))
		print("len d: "+str(len(D)))

		# doing all the metric calculations. anywhere with NaN values is just avoiding a 0 in the denominator error
		if (a+c) > 0:
			prob_det = a/(a+c)
			freq_bias = (a+b)/(a+c)
		else:
			prob_det = 'NaN'
			freq_bias = 'NaN'
		if (b+d) > 0:
			prob_false = b/(b+d)
		else:
			prob_false = 'NaN'
		if ((a+c)*(c+d)+(a+b)*(b+d)) > 0:
			hs_score = (2*((a*d)-(b*c)))/((a+c)*(c+d)+(a+b)*(b+d))
		else:
			hs_score = 'NaN'
			
		pod.append(prob_det)
		pofd.append(prob_false)
		fb.append(freq_bias)
		hss.append(hs_score)

	return hss, rmse, pod, pofd			# can output any of the relevent metrics



def classification(test_df, thresholds, window, station, disc):

	'''calling all the appropriate metric functions, creating a dataframe of the results and then saving the dataframe
		Inputs:
		test_df: all of the prediction and real results dict
		thresholds: list of thresholds to be examined
		window: time window used for evaluation
		'''
	metrics_dict = {}				# creating a dictonary to store the metrics for output
	Metrics = pd.DataFrame()		# for storing the metric data
	Stats = pd.DataFrame()			# storing the RMSE etc.
	# calling the classification calculation function and the precision recall function to get metric scores
	for storm in test_df:
		print(test_df[storm]['real_df'].isnull().values.any())
		storm_year = test_df[storm]['date'].iloc[1].year		# using the storm year to store results
		metrics_dict['{0}_storm'] = {}				# creating a sub dictonary to store the two different data frames
		hss, rmse, pod, pofd = calculating_classification_scores(test_df[storm]['real_df'], thresholds, window)
		Metrics['{0}_HSS'.format(storm_year)] = hss
		Metrics['{0}_POD'.format(storm_year)] = pod
		Metrics['{0}_POFD'.format(storm_year)] = pofd
		Stats['{0}_RMSE'.format(storm_year)] = rmse

	Metrics['thresholds'] = thresholds
	Metrics.set_index('thresholds', drop=True, inplace=True) 		# setting the index on the metrics data to be the thresholds

	Metrics.to_csv('outputs/{0}_metrics_{1}.csv'.format(station, disc))		
	Stats.to_csv('outputs/{0}_stats_{1}.csv'.format(station, disc))		



def plot_outputs(df, stime, etime, station, disc):
	'''plotting the outputs of the model
		Inputs:
		df: testing df with results
		stime: begin time for the plotting of the storm. ONly plotting the main phase of the storm
		etime: end time of the main phase of the storm'''

	storm_year = (datetime.strptime(stime, '%Y-%m-%d %H:%M:%S')).year  			# grabbing the year of the storm for title purposes

	fig = plt.figure(figsize=(60,55))
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)

	ax = fig.add_subplot(111)
	plt.title('{0} Storm'.format(storm_year), fontsize='130')
	ax.margins(x=0)
	plt.plot(df[stime:etime].dBHt, label='real')
	plt.plot(df[stime:etime].predicted, label='predicted')
	ax.set_xticklabels('')
	ax.tick_params(axis='both', labelsize=52)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n %H:%M'))

	plt.savefig('plots/{0}_{1}_{2}.png'.format(str(storm_year), station, disc))


def reindex_and_save_csv(test_dict):
	'''Function that grabs the results dataframe for each storm and 
		re-indexes it to the datetime stamp and then saves the results as a csv.
		Inputs:
		test_dict: the dictonary that contains all of the real data and predicted
								results for each test storm.
	'''
	for i in range(len(test_dict)):
		real_df = test_dict['storm_{0}'.format(i)]['real_df']
		pd.to_datetime(real_df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		real_df.reset_index(drop=True, inplace=True)
		real_df.set_index('Date_UTC', inplace=True, drop=True)
		real_df.index = pd.to_datetime(real_df.index)
		real_df.to_csv('outputs/storm_{0}_{1}_{2}.csv'.format(str(i), station, config['disc']))


def main(path, config, model_config, station, first_time=True):
	'''bringing all of the fun together
		Inputs:
		path: path to the project directory
		config: dict of general configurations for the program
		model_config: dict of model configurations
		first_time: if True, model will be fitted, if False, it will be loaded'''

	print('Entering main...')
	
	df, ace_df = data_prep(path, config['thresholds'], station=station, do_calc=True)  	# calling all of the data prep functuons
	if first_time==True:
		train_dict, scaler = prep_train_data(df, config['test_storm_stime'], config['test_storm_etime'], 
							config['lead'], config['recovery'], model_config['time_history'])  	# calling the training data prep
		with open('models/minmax_{0}_{1}.pkl'.format(str(station), str(config['disc'])), 'wb') as f:			# saving the scaler for future use
			pickle.dump(scaler, f)
	if first_time==False:
		train_dict=pd.DataFrame()																																										# creates empty dataframe to avoid an unidentified variable error
		with open('models/minmax_{0}_{1}.pkl'.format(str(station), str(config['disc'])), 'rb') as f:
			scaler = pickle.load(f)
	
	# calling the test data prep function
	test_dict, n_features = prep_test_data(df, ace_df, config['test_storm_stime'], config['test_storm_etime'],
						model_config['time_history'], scaler, station)
	tf.keras.backend.clear_session()																																					# clears any old saved model parameters.
	MODEL, early_stop = create_CNN_model(model_config, n_features, loss='mse', early_stop_patience=25)			# creating the model
	model = fit_CNN(MODEL, train_dict, model_config, early_stop, station, first_time=first_time, val_size=0.2)  		# fitting the LSTM
	test_dict = making_predictions(model, test_dict)																													# making the predictions

	reindex_and_save_csv(test_dict)		# calling the function that saves the results dataframes as a csv
	
	classification(test_dict, config['thresholds'], config['window'], station, config['disc'])					# calling classification function
	for stime, etime, i in zip(config['test_storm_stime'], config['test_storm_etime'], range(len(test_dict))):			# looping through the storms and plotting the results for each
		plot_outputs(test_dict['storm_{0}'.format(i)]['real_df'], stime, etime, station=station, disc=config['disc'])		# calling the plot_outputs function for each storm



if __name__ == '__main__':

	station = 'OTT'		# the 3 letter code that identifies the station being examined 

	main(projectDir, CONFIG, MODEL_CONFIG, station, first_time=True)

	print('It ran. Good job!')



















