# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
from scipy import stats
from george import kernels
import george
from astropy.io import fits
from functions import *
from glob import glob
from gp_model_params import info
from tqdm import tqdm
import pickle as pkl
from pathlib import Path
from scipy.interpolate import interp1d
import re
from astropy.table import Table
import itertools
from scipy.signal import find_peaks
import warnings




from pathlib import Path
home = str(Path.home())
tp = 'DN'
direc = home+'/Research_data/Bulge_Variables/%s/'%tp
output_plot = 'output/plots/%s/'%tp
output_file = 'output/'
direc_list = np.sort(glob(direc+'*.fits'))


band = 'I'

num_lc = 2000
batch_size = 100

all_peaks = {}

counter = 0

time_offset = 1

for i in tqdm(range(len(direc_list))):

	path = direc_list[i]
	ID = path.split('/')[-1].split('_')[0]
	# if ID != 'OGLE-BLG-DN-0395':
	# 	continue
	all_peaks[ID] = []
	print('Reading light curve %s...'%ID)
	band = 'I'
	if not tp in ['DN', 'CV']:
		period = float(list_periods[list_periods[:, 0] == ID][0][1])
	else:
		period = np.nan


	# Read the light curve data
	period, df_ = read_fits(path, period=period, tp=tp)
	try:
		df = df_[df_.band == band]
		# print(df_.groupby('band').size())
	except AttributeError:
		print('Error: %s band does not exist. Skipping...' % band)

	# Check if sufficient data points exist
	if len(df) < 10:
		print('Error: Found less than 10 data points. Skipping...')
		continue

	df = df.reset_index(drop=True)

	df = df_[df_.band == 'I']

	y_median = np.median(df.m)

	# plt.figure()
	# plt.scatter(df.t.values, -1*df.m.values)
	# plt.savefig(output_plot+'df.png')

	inds = np.where(np.diff(df.t.values)>50)[0]

	# Normalize time axis
	if np.min(df.t) > 0:
		df.t = df.t - np.min(df.t)

	# Reduce the gaps and extend data if necessary
	if tp not in ['DN','CV', 'Be']:
		if (np.max(df.t))<1713:
			df = add_last_point(df, period)
		df_modified = gap_reducer(df, period, info[tp])
		if (np.max(df_modified.t))<1713 and np.isinf(info[tp]['n_phs']):
			df_modified = extend_baseline(df_modified, period)
		while np.max(df_modified.t.values)<1713+period:
			len_0 = np.diff(df_modified.t.values)
			df_modified = gap_reducer(df_modified, period, info[tp])
			len_1 = np.diff(df_modified.t.values)
			if np.max(df_modified.t.values)>1713:
				break
			if np.sum(len_0>50) == np.sum(len_1>50):
				break
	else:
		t_min = np.min(np.diff(df.t.values))
		for i, ind in enumerate(inds):
			time_gap = df.t[ind+1]-df.t[ind]+t_min
			df.loc[df.index[ind+1]:, 't'] -= time_gap
		n_lc = int(max(df.t)//1713.)
		df_modified = df#[df.t<1714]

	# plt.figure()
	# plt.scatter(df_modified.t.values, -1*df_modified.m.values)
	# plt.savefig(output_plot+'df_modified.png')

	mags = df_modified.m.values
	threshold_mag = -0.01 #np.percentile(y_binned, 10)
	baseline_mag = np.percentile(mags, 50) # your quiescent magnitude
	# y_base = np.ones_like(t_base) * baseline_mag + np.random.normal(0, 0.001, size=len(t_base))

	y_mags = mags.copy()
	mask = (mags > threshold_mag+baseline_mag)

	y_mags[mask] = np.ones_like(y_mags[mask]) * baseline_mag #+ np.random.normal(0, 0.001, size=len(y_sim[mask]))

	y_mags = y_mags - np.median(y_mags)
	df_modified.loc[:, 'm'] = y_mags

	# Prepare input for Gaussian Process
	gp = prep_gp(info[tp], period)

	x, y, e = prep_input(df_modified, 
						 tp, 
						 period, 
						 info[tp], 
						 max(df_modified.t.values))# x, y, e are either 
										# the full lc or an intrval 
										# of n phases of them conducted 
										# by repeating a folded phase n times

	# plt.figure()
	# plt.scatter(x, -1*y)
	# plt.savefig(output_plot+'y.png')

	# Bin the data if necessary
	if not np.isinf(info[tp]['n_phs']):
		if len(x)/info[tp]['n_phs']>150:
			if info[tp]['fit_binned']==False:
				info[tp]['fit_binned'] = True
			tot_ideal_len = info[tp]['n_phs']*info[tp]['count_per_bins']
			x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, 
																			  y, 
																			  e, 
																			  info[tp], 
																			  n_bins=tot_ideal_len)
		else:
			info[tp]['fit_binned'] = False
			x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, 
																			  y, 
																			  e, 
																			  info[tp], 
																			  n_bins=np.nan)
	else:
		if len(x)>100:
			info[tp]['fit_binned'] = True
			tot_ideal_len = 50
			x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, 
																			  y, 
																			  e, 
																			  info[tp], 
																			  n_bins=tot_ideal_len)
		else:
			info[tp]['fit_binned'] = False
			x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, 
																		  y, 
																		  e, 
																		  info[tp], 
																		  n_bins=np.nan)
		
		

	x_binned = x[~np.isnan(e)]
	y_binned = y[~np.isnan(e)]
	e_binned = e[~np.isnan(e)]

	# plt.figure()
	# plt.scatter(x_binned, -1*y_binned)
	# plt.savefig(output_plot+'y_binned.png')

	threshold_mag = -0.7 #np.percentile(y_binned, 10)
	baseline_mag = np.percentile(y_binned, 50) # your quiescent magnitude
	# y_base = np.ones_like(t_base) * baseline_mag + np.random.normal(0, 0.001, size=len(t_base))
	y_sim = y_binned.copy()
	mask = (y_binned > threshold_mag)

	y_sim[mask] = np.ones_like(y_sim[mask]) * baseline_mag #+ np.random.normal(0, 0.001, size=len(y_sim[mask]))

	y_sim = y_sim - np.median(y_sim)

	# plt.figure()
	# plt.scatter(x_binned, 
	#         y_sim)

	# # plt.xlim(140, 160)

	# plt.gca().invert_yaxis()
	# plt.savefig(output_plot+'y_sim.png')

	t_peaks, widths = multiscale_peaks(
		x_binned, y_sim,
		scales=[0.1, 0.5, 2, 5, 15],
		min_scales=2,
		merge_window=5,
		min_prominence=0.05,
		min_separation=3.0
	)
	
	# plt.figure()
	# plt.scatter(x_binned, 
	# 		y_sim)

	# # plt.xlim(940, 1200)

	# plt.gca().invert_yaxis()

	# # plt.ylim(-0.01,0.1)
	# for t, w in zip(t_peaks, widths):
	# 	print(f"Peak at t={t:.2f}, FWHM width={w:.2f} days")
		# plt.axvline(t)
	# plt.savefig(output_plot+'all_peaks.png')

	noise_level = 0.01
		
	peak_counter = 0

	for j, (t_peak, w) in enumerate(zip(t_peaks, widths)):
		if w>100:
			print("width is too big.")
			continue
		# if j != len(t_peaks)-1:
		# 	if (np.abs(t_peak - t_peaks[j+1])<5) and (np.abs(w-widths[j+1])<0.1):
		# 		print("Repeated peak=%0.1f, width=%.2f." %(t_peak, w))
		# 		continue
		insert_time = t_peak
		dt_right     = w * 2
		dt_left    = w * 0.8
		dt = 1.5 * w
		ind_peak = (x_binned > t_peak - dt_left) & (x_binned < t_peak + dt_right)
		t = x_binned[ind_peak]

		if len(t)==0:
			ind_peak = (x_binned > t_peak - 2.5*w) & (x_binned < t_peak + 2.5*w)
			t = x_binned[ind_peak]


		# if t[-1]<-0.1:
		# 	ind_peak = (x_binned > t_peak - dt) & (x_binned < t_peak + 2 * dt)
		
		t = x_binned[ind_peak]
		m = y_binned[ind_peak]
		e = e_binned[ind_peak]

		if len(m)<11:
			print("Not enough data points.")
			continue
		
		gp = prep_gp(info[tp], period)
		gp = fit_gp(np.log10((t-min(t))+time_offset), m, e, info[tp], gp)
		x_new = np.linspace(0, max(t)-min(t), int(dt*10))
		y_regular = gp.predict(m, np.log10(x_new+time_offset), return_var=True)[0]

		if len(y_regular) == 0 or not np.any(np.isfinite(y_regular)):
			print("GP prediction returned empty or non-finite array. Skipping peak.")
			continue

		threshold_mag = -0.001
		baseline_mag = 0
		y_regular_tmp = y_regular.copy()
		mask = (y_regular > threshold_mag)
		y_regular_tmp[mask] = np.ones_like(y_regular_tmp[mask]) * baseline_mag

		if len(y_regular_tmp) == 0 or np.max(y_regular_tmp) < -0.1:
			print("Fit not accepted.")
			continue

		if np.std(y_regular_tmp)< 0.3:
			print("std is very low.")
			continue

		if y_regular_tmp[-1]<-0.5 or y_regular_tmp[0]<-0.5:
			print('peak not complete.')
			continue
		interp_y1 = interp1d(x_new, y_regular_tmp, kind='linear', bounds_error=False, fill_value=np.nan)

		all_peaks[ID].append({
								"gp": gp,
								"xfit": np.log10((t-min(t))+time_offset),
								"yfit": m,
								"x": t,
								"y": m,
								"x_new":x_new,
								"y_new":y_regular_tmp,
								"interpol":interp_y1,
								"t_peak": t_peak
								})
		peak_counter += 1

		plt.figure()
		plt.scatter(t-min(t), m)
		plt.plot(x_new, y_regular_tmp, 'r')
		plt.text(0.05, 0.2, "min=%0.1f, std=%0.3f"%(np.max(y_regular_tmp), np.std(y_regular_tmp)), transform=plt.gca().transAxes)
		plt.gca().invert_yaxis()
		plt.savefig(output_plot + '_' + ID + '_%i.png'%peak_counter)

pkl.dump(all_peaks, 
		 open(output_file+'%s_peaks_bank.pkl'%tp,
			  'wb'))

			
	
