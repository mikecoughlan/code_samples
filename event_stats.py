######################################################################################################################
#
# '~/projects/single_station_event_counting/event_stats.py'
# File that takes data from solar wind (OMNI database), F 10.7 data as a solar activity indicator, and ground 
# magnetometer stations to first detect events, as defined by dB_H/dt exceeding a given threshold value, and then
# examining a variety of variables in an attempt to find a coorelation. Also seperates events based on latitude 
# to examine differences in the coorelations between solar wind parameters and dB_H/dt events on the earth's surface. 
# 
#
######################################################################################################################


import pandas as pd
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import inspect
import time
import random
import matplotlib.dates as mdates


percentile = 99.97	# data percentile to be examined
ProjectDir = '~/projects/single_station_event_counting/' # defining the project directory

stations = ['THL', 'BOU', 'HRN', 'RES', 'MEA', 'SIT', 'VIC', 'NEW', 'LER', 
			'VAL', 'HAD', 'ESK', 'MAB', 'DOU', 'CLF', 'OTT', 'WNG', 
			'THY', 'BEL', 'FUR', 'NGK', 'FRD', 'FCC', 'TRO', 
			'ABK', 'SOD', 'NUR', 'BRW', 'CMO', 'ASP', 'CNB', 'CZT', 
			'DRV', 'GUA', 'HER', 'HON', 'KAK', 'MBO', 'MMB', 'NCK', 
			'SBA', 'TUC', 'KNY', 'MAW'] # stations to be examined. Were choosen amongst SUPERMAG stations because of data continuity


def data_prep(path, station, ssndf, omnidf, do_calc=True):
	''' Preparing the magnetometer data for the other functions
		Inputs:
		path: the file path to the project directory
		station: the magnetometer station to be examined. Comes from the list of stations
		that will be looped over. 
		ssndf: pandas Series object containing daily information on the solar radio flux at
				10.7 cm in W * m^-2 * Hz^-1
		omnidf: data frame with omni data including some calculated values from quiver_map_plots.
	'''
	print('preparing {0} data...'.format(station))

	if do_calc:
		df = pd.read_csv(path+'data/{0}.csv'.format(station)) # loading the station data.
		df['dN'] = df['N'].diff(1) # creates the dN column
		df['dE'] = df['E'].diff(1) # creates the dE column
		df['dBHt'] = np.sqrt(((df['N'].diff(1))**2)+((df['E'].diff(1))**2)) # creates the combined dB/dt column
		df['direction'] = (np.arctan2(df['dN'], df['dE']) * 180 / np.pi)	# calculates the angle of dB/dt
		df['year'] = pd.DatetimeIndex(df['Date_UTC']).year		# making df column for the year
		df['month'] = pd.DatetimeIndex(df['Date_UTC']).month	# making df column for the months
		if df['MLAT'].dtypes == 'object':
			df['MLAT'] = df['MLAT'].replace(['84.w2'], 84.72)
		df['MLAT'] = df['MLAT'].astype('float64')

		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index)

		df = pd.concat([df, ssndf], axis=1, ignore_index=False)
		df = pd.concat([df, omnidf], axis=1, ignore_index=False)
		df['F107'] = df.F107.interpolate(method='linear')			# time resoultion is larger for F10.7, so we interpolate
		features = ['dBHt', 'dN', 'dE', 'GEOLON', 'GEOLAT', 'MLAT', 'MLT', 'year', 'month', 'direction', 
					'F107', 'B_Total', 'BZ_GSM', 'flow_speed', 'Vx', 'proton_density', 'AE_INDEX', 'Beta', 
					'SYM_H', 'n_std', 'Vx_std', 'BZ_std'] # listing the features we want to keep
		datum = df.dropna()
		datum = datum[features][1:] # drops all features not in the features list above and drops the first row because of the derivatives
		datum.to_csv('data/{0}_prepared.csv'.format(station))			# saving file so calculations don't have to be re-done
		datum.reset_index(drop=True, inplace=True)

	if not do_calc:
		datum = pd.read_csv('data/{0}_prepared.csv'.format(station))
		datum.reset_index(drop=True, inplace=True)

	return datum

def F107_prep(path):
	'''Loading and preparing the F10.7 data for examination. Only has path input for finding data.'''

	print('Getting F10.7 data...')
	df = pd.read_csv(path + 'data/F107.csv')
	df.loc[df['F107'] >= 999, 'F107'] = np.nan		# missing data is set to 999.9 in database, replacing with nan
	df['F107'] = df.F107.interpolate(method='linear')	# interpolating over the nan values
	pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('date', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	F107 = df['F107']

	return F107


def omni_prep(path, std_len=10, do_calc=True):

'''Preparing the omnidata for plotting.
	Inputs:
	path: path to project directory
	sdate: storms start date
	edate: storm end date
	std_len: lookback length over which the standard deviation is calculated. i.e. for default
			std_len=30 the standard deviation of a parameter at time t will be calculated from
			df[param][t-30:t].
	do_calc: (bool) is true if the calculations need to be done, false if this is not the first time
			running this specific configuration and a csv has been saved. If this is the case the csv
			file will be loaded.
'''

	if do_calc:

		df = pd.read_csv(path+'data/omni.csv') # loading the omni data

		df.reset_index(drop=False, inplace=True) # reseting the index so its easier to work with integer indexes

		features = ['Epoch', 'B_Total',
		   			'BZ_GSM', 'flow_speed', 'Vx',
		   			'proton_density', 'AE_INDEX', 'Beta', 'SYM_H'] # defining the features to keep from the dataframe
		df = df[features] # dropping all features not in above feature list
		df['dn/dt'] = df['proton_density'].diff(1) # defining the time derivative in the proton density
		df.loc[df['Beta'] >= 2, 'Beta'] = np.nan		# replacing missing values

		# looping through dataframe to add columns
		for i in range(len(df)):
			# does calculations for the first i minutes and assignes the calulations to ith row
			if i < std_len:			# calculates the standard deviation over a smaller range for the first few time steps
				bz = df['BZ_GSM'][:i].std() # calculating the std_dev of the previous i minutes of BZ_GSM
				vx = df['Vx'][:i].std() # calculating the std_dev of the previous i minutes of Vx
				n = df['proton_density'][:i].std() # calculating the std_dev of the previous i minutes of proton density
				df.at[i, 'BZ_std'] = bz # assinges std_dev BZ_GSM to ith row
				df.at[i, 'Vx_std'] = vx # assinges std_dev Vx to ith row
				df.at[i, 'n_std'] = n # assinges std_dev proton density to ith row
			# for the rest of the dataframe calculates the standard devs for the previous t minutes
			else:
				bz = df['BZ_GSM'][i-std_len:i].std()
				vx = df['Vx'][i-std_len:i].std()
				n = df['proton_density'][i-std_len:i].std()
				df.at[i, 'BZ_std'] = bz
				df.at[i, 'Vx_std'] = vx
				df.at[i, 'n_std'] = n

		# fills the empty dataframe objects in these columns with value of 0
		df['n_std'].fillna(0, inplace=True) 
		df['BZ_std'].fillna(0, inplace=True)
		df['Vx_std'].fillna(0, inplace=True)
		df = df.dropna() # shouldn't be any empty rows after that but in case there is we drop them here

		# reassign the datetime object as the index
		pd.to_datetime(df['Epoch'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Epoch', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index)

		df.to_csv(path+'data/omni_csv_calcs.csv')

	if not do_calc:

		df = pd.read_csv(path+'data/omni_csv_calcs.csv') # loading the omni data

		# reassign the datetime object as the index
		pd.to_datetime(df['Epoch'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Epoch', inplace=True, drop=True)
		df.index = pd.to_datetime(df.index)


	return df

def creating_percentiles(path, stations, percentile):
'''takes the data from the stations, averages the MLAT and finds the percentile in nT/min.
	INPUTS:
	path: file path to project directory.
	stations: list of stations to be examined.
	percentile(float): data percentile to be examined i.e. 99.0 means 99th percentile.
'''

	print('Creating mlat and thresholds using percentile...')
	MLAT, thresholds, df_list = [], [], []
	omni = omni_prep(path, do_calc=False)
	F107 = F107_prep(path)
	for station in stations:
		station = data_prep(path, station, F107, omni, do_calc=True)	# doing data prep on the stations to create columns
		MLAT.append(np.abs(station['MLAT'].mean()))		# because the MLAT changes with time we average it for plotting
		dBHt = station['dBHt'].to_numpy()	# making a numpy array to calculate percentile
		thresholds.append(np.percentile(dBHt, percentile).round(decimals=2))	# logs the percentile of the dBHt for this station
		df_list.append(station)		# adding the station data to a list of dfs for further manipulations. May have to adjust if mem issue

	return MLAT, thresholds, df_list


def perc_v_lat(mlat, P_thresholds, perc):
	'''plots the mean MLAT vs the percentile to see how the top percentiles change with latitude.
		INPUTS:
		mlat: averaged MLAT positions for each examined station.
		P_thresholds: perc value for each station in nT/min.
		perc: threshold percentile being examined.
	'''
	print('Creating threshold vs. MLAT plot...')
	fig = plt.figure(figsize=(60,20))
	plt.subplots_adjust(bottom=-0.9, top=0.9, left=0.1, right=0.86, hspace=0.05)

	ax = fig.add_subplot(1,1,1)
	ax.scatter(mlat,P_thresholds, s=200)
	ax.set_title('P{0}'.format(perc), fontsize='52')
	ax.set_xlabel('Average MLAT', fontsize='45')
	ax.set_ylabel('Thresholds (nT/min)', fontsize='45')
	ax.margins(x=0, y=0)
	ax.grid()
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')

	plt.savefig('plots/rogers_plots/percentile_vs_mlat_P{0}.png'.format(perc), 
					bbox_inches='tight', pad_inches=1)



def extracting_events(df_list, P_thresholds):
	'''For each station we extract the events that exceed that stations perc value. 
		INPUTS:
		df_list: the preprepared list of full data frames for each station. 
		P_thresholds: the list of thresholds that represent the nth percentile for that station.
	'''
	print('Extracting events...')
	data = list()	# initalizing a list
	for df, thresh in zip (df_list, P_thresholds):
		datum = df[(df['dBHt'] >= thresh)]	# extracting the events that exceed the thresholds and creating a new df
		datum.reset_index(inplace=True, drop=False)	# resetting the index of the df
		data.append(datum)	# adding the df to the list
	df = pd.concat(data, ignore_index=True)	# concatinating the dfs along the index

	return df



def extract_and_seperate_hem_and_lat(df_list, P_thresholds):
'''seperates the data into northern and southern hemisphere dataframes. And then into low, mid, and high latitude stations.
	INPUTS:
	df_list: list of dataframes. One for each station being examined.
	P_threhsolds: the list of threshold values that coorespond to the percentile for the stations
	'''

	# create list to store seperated dataframes
	n_high_df, n_mid_df, n_low_df, s_high_df, s_mid_df, s_low_df = list(), list(), list(), list(), list(), list() 

	for df, thresh in zip(df_list, P_thresholds):
		datum = df[(df['dBHt'] >= thresh)]	# extracting the events that exceed the thresholds and creating a new df
		
		north = datum[(datum['GEOLAT'] > 0)]	# getting northern hemisphere stations
		south = datum[(datum['GEOLAT'] < 0)]	# getting southern hemisphere stations

		# dividing each hemisphere into low, mid, and high sections for examination
		n_low = north[(north['GEOLAT'] < 40)]
		n_mid = north[(north['GEOLAT'] >= 40) & (north['GEOLAT'] <= 70)]
		n_high = north[(north['GEOLAT'] > 70)]

		s_low = south[(south['GEOLAT'] > -40)]
		s_mid = south[(south['GEOLAT'] <= -40) & (south['GEOLAT'] >= -70)]
		s_high = south[(south['GEOLAT'] < -70)]

		n_low.reset_index(inplace=True, drop=False)
		n_mid.reset_index(inplace=True, drop=False)
		n_high.reset_index(inplace=True, drop=False)

		s_low.reset_index(inplace=True, drop=False)
		s_mid.reset_index(inplace=True, drop=False)
		s_high.reset_index(inplace=True, drop=False)

		n_low_df.append(n_low)
		n_mid_df.append(n_mid)
		n_high_df.append(n_high)
		s_low_df.append(s_low)
		s_mid_df.append(s_mid)
		s_high_df.append(s_high)
		
	nlowdf = pd.concat(n_low_df, ignore_index=True)
	nmiddf = pd.concat(n_mid_df, ignore_index=True)
	nhighdf = pd.concat(n_high_df, ignore_index=True)
	slowdf = pd.concat(s_low_df, ignore_index=True)
	smiddf = pd.concat(s_mid_df, ignore_index=True)
	shighdf = pd.concat(s_high_df, ignore_index=True)

	return nlowdf, nmiddf, nhighdf, slowdf, smiddf, shighdf	# concatinating the dfs along the index


def extract_and_seperate_lat(df_list, P_thresholds):
'''seperates the data into different latitides for plotting.
	INPUTS:
	df_list: list of dataframes. One for each station being examined.
	P_threhsolds: the list of threshold values that coorespond to the percentile for the stations
	'''

	# create list to store seperated dataframes
	high_df, mid_df, low_df = list(), list(), list() 

	for df, thresh in zip (df_list, P_thresholds):
		datum = df[(df['dBHt'] >= thresh)]	# extracting the events that exceed the thresholds and creating a new df
		
		datum['GEOLAT'] = datum['GEOLAT'].abs()	# getting southern hemisphere stations
		datum['MLAT'] = datum['MLAT'].abs()	# changing to absoulte mlat for plotting

		# dividing each hemisphere into low mid and high sections for examination
		low = datum[(datum['GEOLAT'] < 40)]
		mid = datum[(datum['GEOLAT'] >= 40) & (datum['GEOLAT'] <= 70)]
		high = datum[(datum['GEOLAT'] > 70)]

		low.reset_index(inplace=True, drop=False)
		mid.reset_index(inplace=True, drop=False)
		high.reset_index(inplace=True, drop=False)

		low_df.append(low)
		mid_df.append(mid)
		high_df.append(high)
	
	# concatinating the dfs along the index
	lowdf = pd.concat(low_df, ignore_index=True)
	middf = pd.concat(mid_df, ignore_index=True)
	highdf = pd.concat(high_df, ignore_index=True)

	return lowdf, middf, highdf	


def var_sep(df_list, P_thresholds, param, max_range, min_range):
	'''Takes the seperated low, mid and high latitude dataframes, and seperates them based on 
		the given variable. Will seperate into below the min range, between the min and max range,
		and above the max range.
		INPUTS:
		df_list: the preprepared list of full data frames for each station. 
		P_thresholds: the list of thresholds that represent the nth percentile for that station.
		param: the variable used to seperate the dataframes.
		max_range: (int or float) upper bound on the variable to seperate.
		min_range: (int or float) lower bound on teh variable to seperate.
	'''

	lowdf, middf, highdf = extract_and_seperate_lat(df_list, P_thresholds)

	low_min = lowdf[(lowdf[param] < min_range)]		# extracts the lower portion of the variable	
	low_range = lowdf[(lowdf[param] >= min_range) & (lowdf[param] <= max_range)]			# extracts the central range of the variable
	low_max = lowdf[(lowdf[param] > max_range)]				# the upper range

	mid_min = middf[(middf[param] < min_range)]
	mid_range = middf[(middf[param] >= min_range) & (middf[param] <= max_range)]
	mid_max = middf[(middf[param] > max_range)]

	high_min = highdf[(highdf[param] < min_range)]
	high_range = highdf[(highdf[param] >= min_range) & (highdf[param] <= max_range)]
	high_max = highdf[(highdf[param] > max_range)]

	low_min.reset_index(inplace=True, drop=True)
	low_range.reset_index(inplace=True, drop=True)
	low_max.reset_index(inplace=True, drop=True)

	mid_min.reset_index(inplace=True, drop=True)
	mid_range.reset_index(inplace=True, drop=True)
	mid_max.reset_index(inplace=True, drop=True)

	high_min.reset_index(inplace=True, drop=True)
	high_range.reset_index(inplace=True, drop=True)
	high_max.reset_index(inplace=True, drop=True)

	return low_min, low_range, low_max, mid_min, mid_range, mid_max, high_min, high_range, high_max



def hist2D(df_list, P_thresholds, xparam, yparam, xbins, ybins):
	'''Making 2d histogram/heatmap plots. Defaults to MLT on the x-axis.
		INPUTS:
		df: dataframe containing the events.
		xparam(yparam): parameter for the x(y) axis.
		xbins(ybins): number of bins for the x(y) direction.
	'''
	print('Creating {0} vs. {1} historgram...'.format(yparam, xparam))

	df = extracting_events(df_list, P_thresholds)

	df['MLAT'] = df['MLAT'].abs()			# will combine northern and southern hemispheres
	x = df[xparam].to_numpy()				# puts the x and y params into numpy arrays
	y = df[yparam].to_numpy()

	fig = plt.figure(figsize=(60,30))
	plt.subplots_adjust(bottom=-0.9, top=0.9, left=0.1, right=0.86, hspace=0.05)

	ax = fig.add_subplot(1,1,1)
	h = ax.hist2d(x, y, bins=[xbins,ybins], cmap='jet')
	ax.set_title('{0} vs. {1}'.format(yparam, xparam), fontsize='52')
	ax.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax.set_ylim(0,90)
	ax.margins(x=0, y=0)
	ax.grid(linewidth=5)
	plt.xticks([0,3,6,9,12,15,18,21,24])		# 3 hour incriments for labeling, allows us to distinguish important magnetospheric trends
	cbar = plt.colorbar(h[3], ax=ax)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')

	plt.savefig('plots/{0}_vs_{1}_hist.png'.format(yparam, xparam), 
					bbox_inches='tight', pad_inches=1)


def sep_hem_and_lat_hist2D(df_list, P_thresholds, xparam, yparam, xbins, ybins):
	'''Making 2d histogram/heatmap plots.
		INPUTS:
		df: dataframe containing the events.
		xparam(yparam): parameter for the x(y) axis.
		xbins(ybins): number of bins for the x(y) direction.
	'''
	print('Creating {0} vs. {1} historgram...'.format(yparam, xparam))

	n_low_df, n_mid_df, n_high_df, s_low_df, s_mid_df, s_high_df = extract_and_seperate_hem_and_lat(df_list, P_thresholds)

	# putting all of the seperated parameters into numpy arrays for plotting.

	n_low_x = n_low_df[xparam].to_numpy()
	n_low_y = n_low_df[yparam].to_numpy()

	n_mid_x = n_mid_df[xparam].to_numpy()
	n_mid_y = n_mid_df[yparam].to_numpy()

	n_high_x = n_high_df[xparam].to_numpy()
	n_high_y = n_high_df[yparam].to_numpy()

	s_low_x = s_low_df[xparam].to_numpy()
	s_low_y = s_low_df[yparam].to_numpy()

	s_mid_x = s_mid_df[xparam].to_numpy()
	s_mid_y = s_mid_df[yparam].to_numpy()

	s_high_x = s_high_df[xparam].to_numpy()
	s_high_y = s_high_df[yparam].to_numpy()

	fig = plt.figure(figsize=(60,30))
	plt.subplots_adjust(bottom=-0.9, top=0.9, left=0.1, right=0.86, hspace=0.1)

	ax1 = fig.add_subplot(6,3,1)
	h = ax1.hist2d(n_low_x, n_low_y, bins=[xbins,ybins], cmap='jet')
	ax1.set_title('Low Latitude (<40)'.format(yparam, xparam), fontsize='68')
	ax1.margins(x=0, y=0)
	ax1.grid()
	ax1.annotate('Northern Hemisphere', xy=(0, -1),
                xycoords=ax1.yaxis.label, rotation=90,
                fontsize='40')
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax1)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax2 = fig.add_subplot(6,3,2)
	h = ax2.hist2d(n_mid_x, n_mid_y, bins=[xbins,ybins], cmap='jet')
	ax2.set_title('Mid Latitude (40-70)'.format(yparam, xparam), fontsize='68')
	ax2.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax2.margins(x=0, y=0)
	ax2.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax2)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax3 = fig.add_subplot(6,3,3)
	h = ax3.hist2d(n_high_x, n_high_y, bins=[xbins,ybins], cmap='jet')
	ax3.set_title('High Latitude (>70)'.format(yparam, xparam), fontsize='68')
	ax3.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax3.margins(x=0, y=0)
	ax3.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax3)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax4 = fig.add_subplot(6,3,4)
	h = ax4.hist2d(s_low_x, s_low_y, bins=[xbins,ybins], cmap='jet')
	ax4.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax4.margins(x=0, y=0)
	ax4.grid()
	ax4.annotate('Southern Hemisphere', xy=(0, -1),
                xycoords=ax4.yaxis.label, rotation=90,
                fontsize='40')
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax4)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax5 = fig.add_subplot(6,3,5)
	h = ax5.hist2d(s_mid_x, s_mid_y, bins=[xbins,ybins], cmap='jet')
	ax5.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax5.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax5.margins(x=0, y=0)
	ax5.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax5)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax6 = fig.add_subplot(6,3,6)
	h = ax6.hist2d(s_high_x, s_high_y, bins=[xbins,ybins], cmap='jet')
	ax6.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax6.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax6.margins(x=0, y=0)
	ax6.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax6)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')

	plt.savefig('plots/{0}_vs_{1}_sep_lats_hist_9997.png'.format(yparam, xparam), 
					bbox_inches='tight', pad_inches=1)


def sep_lat_hist2D(df_list, P_thresholds, xparam, yparam, xbins, ybins, cm):
	'''Making 2d histogram/heatmap plots.
		INPUTS:
		df: dataframe containing the events.
		xparam(yparam): parameter for the x(y) axis.
		xbins(ybins): number of bins for the x(y) direction.
		cm: int, float or None. sets bin max limit on histograms.
	'''
	print('Creating {0} vs. {1} historgram...'.format(yparam, xparam))

	low_df, mid_df, high_df = extract_and_seperate_lat(df_list, P_thresholds)

	# putting all of the seperated parameters into numpy arrays for plotting.

	low_x = low_df[xparam].to_numpy()
	low_y = low_df[yparam].to_numpy()

	mid_x = mid_df[xparam].to_numpy()
	mid_y = mid_df[yparam].to_numpy()

	high_x = high_df[xparam].to_numpy()
	high_y = high_df[yparam].to_numpy()

	fig = plt.figure(figsize=(60,30))
	plt.subplots_adjust(bottom=-0.9, top=0.9, left=0.1, right=0.86, hspace=0.1)

	ax1 = fig.add_subplot(3,3,1)
	h = ax1.hist2d(low_x, low_y, bins=[xbins,ybins], cmap='jet', cmax=cm)
	ax1.set_title('Low Latitude (<40)'.format(yparam, xparam), fontsize='68')
	ax1.margins(x=0, y=0)
	ax1.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax1)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax2 = fig.add_subplot(3,3,2)
	h = ax2.hist2d(mid_x, mid_y, bins=[xbins,ybins], cmap='jet', cmax=cm)
	ax2.set_title('Mid Latitude (40-70)'.format(yparam, xparam), fontsize='68')
	ax2.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax2.margins(x=0, y=0)
	ax2.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax2)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax3 = fig.add_subplot(3,3,3)
	h = ax3.hist2d(high_x, high_y, bins=[xbins,ybins], cmap='jet', cmax=cm)
	ax3.set_title('High Latitude (>70)'.format(yparam, xparam), fontsize='68')
	ax3.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax3.margins(x=0, y=0)
	ax3.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax3)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	plt.savefig('plots/rogers_plots/{0}_vs_{1}_hist.png'.format(yparam, xparam), 
					bbox_inches='tight', pad_inches=1)


def sep_var_plot(df_list, P_thresholds, xparam, yparam, xbins, ybins, param, max_range, min_range):
	'''Making 2d histogram/heatmap plots for each seperate latitudes 
		and with different variable ranges.
		INPUTS:
		df: dataframe containing the events.
		xparam(yparam): parameter for the x(y) axis.
		xbins(ybins): number of bins for the x(y) direction.
		param: the variable used to seperate the dataframes.
		max_range: (int or float).
		min_range: (int or float).'''

	print('Creating {0} vs. {1} historgram...'.format(yparam, xparam))

	low_min, low_range, low_max, mid_min, mid_range, mid_max, high_min, high_range, high_max = var_sep(df_list, P_thresholds, param, max_range, min_range)

	# putting all of the seperated parameters into numpy arrays for plotting.

	low_min_x = low_min[xparam].to_numpy()
	low_min_y = low_min[yparam].to_numpy()

	low_range_x = low_range[xparam].to_numpy()
	low_range_y = low_range[yparam].to_numpy()

	low_max_x = low_max[xparam].to_numpy()
	low_max_y = low_max[yparam].to_numpy()

	mid_min_x = mid_min[xparam].to_numpy()
	mid_min_y = mid_min[yparam].to_numpy()

	mid_range_x = mid_range[xparam].to_numpy()
	mid_range_y = mid_range[yparam].to_numpy()

	mid_max_x = mid_max[xparam].to_numpy()
	mid_max_y = mid_max[yparam].to_numpy()

	high_min_x = high_min[xparam].to_numpy()
	high_min_y = high_min[yparam].to_numpy()

	high_range_x = high_range[xparam].to_numpy()
	high_range_y = high_range[yparam].to_numpy()

	high_max_x = high_max[xparam].to_numpy()
	high_max_y = high_max[yparam].to_numpy()

	fig = plt.figure(figsize=(60,30))
	plt.subplots_adjust(bottom=-0.9, top=0.9, left=0.1, right=0.86, hspace=0.1)

	ax1 = fig.add_subplot(9,3,1)
	h = ax1.hist2d(low_min_x, low_min_y, bins=[xbins,ybins], cmap='jet')
	ax1.set_title('Low Latitude (<40)'.format(yparam, xparam), fontsize='68')
	ax1.margins(x=0, y=0)
	ax1.grid()
	ax1.annotate('{0} < {1}'.format(param, min_range), xy=(-0.5, -1),
                xycoords=ax1.yaxis.label, rotation=90,
                fontsize='40')
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax1)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax2 = fig.add_subplot(9,3,2)
	h = ax2.hist2d(mid_min_x, mid_min_y, bins=[xbins,ybins], cmap='jet')
	ax2.set_title('Mid Latitude (40-70)'.format(yparam, xparam), fontsize='68')
	ax2.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax2.margins(x=0, y=0)
	ax2.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax2)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax3 = fig.add_subplot(9,3,3)
	h = ax3.hist2d(high_min_x, high_min_y, bins=[xbins,ybins], cmap='jet')
	ax3.set_title('High Latitude (>70)'.format(yparam, xparam), fontsize='68')
	ax3.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax3.margins(x=0, y=0)
	ax3.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax3)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax4 = fig.add_subplot(9,3,4)
	h = ax4.hist2d(low_range_x, low_range_y, bins=[xbins,ybins], cmap='jet')
	ax4.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax4.margins(x=0, y=0)
	ax4.grid()
	ax4.annotate('{0} < {1} < {2}'.format(min_range, param, max_range), xy=(0, -1),
                xycoords=ax4.yaxis.label, rotation=90,
                fontsize='40')
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax4)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax5 = fig.add_subplot(9,3,5)
	h = ax5.hist2d(mid_range_x, mid_range_y, bins=[xbins,ybins], cmap='jet')
	ax5.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax5.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax5.margins(x=0, y=0)
	ax5.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax5)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax6 = fig.add_subplot(9,3,6)
	h = ax6.hist2d(high_range_x, high_range_y, bins=[xbins,ybins], cmap='jet')
	ax6.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax6.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax6.margins(x=0, y=0)
	ax6.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax6)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax7 = fig.add_subplot(9,3,7)
	h = ax7.hist2d(low_max_x, low_max_y, bins=[xbins,ybins], cmap='jet')
	ax7.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax7.margins(x=0, y=0)
	ax7.grid()
	ax7.annotate('{0} > {1}'.format(param, min_range), xy=(0, -1),
                xycoords=ax7.yaxis.label, rotation=90,
                fontsize='40')
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax7)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax8 = fig.add_subplot(9,3,8)
	h = ax8.hist2d(mid_max_x, mid_max_y, bins=[xbins,ybins], cmap='jet')
	ax8.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax8.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax8.margins(x=0, y=0)
	ax8.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax8)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	ax9 = fig.add_subplot(9,3,9)
	h = ax9.hist2d(high_max_x, high_max_y, bins=[xbins,ybins], cmap='jet')
	ax9.set_xlabel('{0}'.format(xparam), fontsize='45')
	ax9.set_ylabel('{0}'.format(yparam), fontsize='45')
	ax9.margins(x=0, y=0)
	ax9.grid()
	plt.xticks([0,3,6,9,12,15,18,21,24])
	cbar = plt.colorbar(h[3], ax=ax9)
	cbar.ax.tick_params(labelsize=36)
	plt.tick_params(axis='x', labelsize='36')
	plt.tick_params(axis='y', labelsize='36')


	plt.savefig('plots/{0}_vs_{1}_{2}_var.png'.format(yparam, xparam, param), 
					bbox_inches='tight', pad_inches=1)



def main(path, stations, perc):
'''Puts all the functions together.
	INPUTS:
	path: path to the main project directory.
	stations: list of stations to be examined.
	perc: percentile to be examined, ie. 99.9th percentile for each station.
	'''

	print('Entering main...')

	params = ['direction', 'F107', 'SYM_H', 'Beta', 'BZ_std', 'Vx', 'n_std', 'MLAT']	# parameters selected for seperation and plotting
	x_bins = [48, 24, 24, 24, 24, 24, 24, 24]			# x bins for the parameters in params
	y_bins = [72, 35, 90, 20, 30, 200, 20, 45]			# y bins for the parameters in params
	cmax = [None, None, None, 5, 10, None, 10, None]	# cooresponding camx for the 2D hist plots

	mlat, P_thresholds, df_list = creating_percentiles(path, stations, perc)	# finding the dB_H/dt values cooresponding to the percentiels for each station
	perc_v_lat(mlat, P_thresholds, perc)	# plotting the thresholds vs the magnetic latitude
	hist2D(df_list, P_thresholds, 'MLT', 'MLAT', 24, 45)	# making a 2D histogram
	sep_hem_and_lat_hist2D(df_list, P_thresholds, 'MLT', 'direction', 48, 72)	# making a 2D histogram
	
	for param, xbin, ybin, cm in zip(params, x_bins, y_bins, cmax):
		sep_lat_hist2D(df_list, P_thresholds, 'MLT', param, xbin, ybin, cm)	# making a 2D histogram

	sep_var_plot(df_list, P_thresholds, 'MLT', 'direction', 24, 72, 'BZ_GSM', 2, -2)	# another 2D histogram



if __name__ == '__main__':

	main(ProjectDir, stations, percentile)
	
	print('It ran. Good job!')




