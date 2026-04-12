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



from pathlib import Path
home = str(Path.home())

tp = 'Flares_TESS'

# Here we read Roman cadences for the Galactic Bulge
Roman_sampling_short = np.load('lc_example/roman_times_shortcadence.npy')
Roman_sampling_long = np.load('lc_example/roman_times_longcadence.npy')
Roman_sampling_long2 = np.load('lc_example/roman_times_longcadence2.npy')
Roman_sampling_short_min = min(Roman_sampling_short)
Roman_sampling_short = Roman_sampling_short - Roman_sampling_short_min
Roman_sampling_long_min = min(Roman_sampling_long)
Roman_sampling_long = Roman_sampling_long - Roman_sampling_long_min
Roman_sampling_long2_min = min(Roman_sampling_long2)
Roman_sampling_long2 = Roman_sampling_long2 - Roman_sampling_long2_min
Roman_max_short_time = max(Roman_sampling_short)

t_base = Roman_sampling_short
t_base_long = Roman_sampling_long
t_base_long2 = Roman_sampling_long2


path = home+'/Research_data/Bulge_Variables/Flares_TESS/'
output_plot = 'output/plots/FL/'
output_file = 'output/flares/'

with open(output_file + 'flare_baseline_bank.pkl', 'rb') as file:
	baseline_bank = pkl.load(file)

with open(output_file + 'flare_gp_fit_bank.pkl', 'rb') as file:
	flare_bank = pkl.load(file)

band = 'TESS'

num_lc = 2000
batch_size = 100

make_flat = ['117874959',
			 '158624542',
			 '197829751']

all_flares = len(list(flare_bank.keys()))
flare_keys = list(flare_bank.keys())
all_baselines = len(list(baseline_bank.keys()))

all_lc = {}
counter = 0

for i in tqdm(range(num_lc)):
	
	tic_id = np.random.choice(list(baseline_bank.keys()), size=1)[0]
	n_flares = int(np.random.choice(np.linspace(0, all_flares-1, all_flares-1, dtype=int), size=1)[0])
	flare_inds = np.random.choice(flare_keys, size=n_flares)



	if tic_id in make_flat:

		ID = str(i)+'_flat_'+str(n_flares)
		all_lc[ID] = {}

		t_regular = baseline_bank[tic_id]['%s_band_Roman_time_regular_sampling'%band]
		
		baseline_mag = np.median(baseline_bank[tic_id]['%s_band_Roman_m_official_sampling_short'%band])   # your quiescent magnitude
		y_base = np.ones_like(t_base) * baseline_mag + np.random.normal(0, 0.001, size=len(t_base))
		y_base_long = np.ones_like(t_base_long) * baseline_mag + np.random.normal(0, 0.001, size=len(t_base_long))
		y_base_long2 = np.ones_like(t_base_long2) * baseline_mag + np.random.normal(0, 0.001, size=len(t_base_long2))
		y_base_regular = np.ones_like(t_regular) * baseline_mag + np.random.normal(0, 0.001, size=len(t_regular))
	
	else:
		ID = str(i)+'_'+tic_id+'_'+str(n_flares)
		all_lc[ID] = {}
		y_base = baseline_bank[tic_id]['%s_band_Roman_m_official_sampling_short'%band]
		y_base_long = baseline_bank[tic_id]['%s_band_Roman_m_official_sampling_long'%band]
		y_base_long2 = baseline_bank[tic_id]['%s_band_Roman_m_official_sampling_long2'%band]
		y_base_regular = baseline_bank[tic_id]['%s_band_Roman_m_regular_sampling'%band]
		t_regular = baseline_bank[tic_id]['%s_band_Roman_time_regular_sampling'%band]
		
	print(ID)
	y_sim = y_base.copy()
	y_sim_long = y_base_long.copy()
	y_sim_long2 = y_base_long2.copy()
	y_sim_regular = y_base_regular.copy()
	
	flare_times = np.random.choice(t_base, size=n_flares)
	
	for j, flare_ind in enumerate(flare_inds):
		# print(flare_ind)
		
		flare = flare_bank[flare_ind]
		gp = flare["gp"]
		xfit_flare = flare["xfit"]
		xflare = flare["x"]
		flare_tmax = flare["x"][np.argmin(flare["yfit"])]
		tmin_flare = flare_tmax - xflare.min()
		tmax_flare = xflare.max() - flare_tmax   
		interp_y1 = interp1d(flare["x_new"], flare["y_new"], kind='linear', bounds_error=False, fill_value=np.nan)
	
		t0 = flare_times[j]
		mask = (t_base > t0 - tmin_flare) & (t_base < t0 + tmax_flare)
		mask_regular = (t_regular > t0 - tmin_flare) & (t_regular < t0 + tmax_flare)


		
		
		
		
		
		
		if np.sum(mask)!=0:
			t_local = t_base[mask] - t_base[mask].min() + tmin_flare
			flare_mag = interp_y1(t_local)
			y_sim[mask] += flare_mag-max(flare_mag)

		if np.sum(mask_regular) != 0:
			t_local_regular = t_regular[mask_regular] - t_regular[mask_regular].min() + tmin_flare

			flare_mag_regular = interp_y1(t_local_regular)
			y_sim_regular[mask_regular] += flare_mag_regular-max(flare_mag_regular)

		mask_long = (t_base_long > t0 - tmin_flare) & (t_base_long < t0 + tmax_flare)
		if np.sum(mask_long)!=0:
			t_local_long = t_base_long[mask_long] - t_base_long[mask_long].min() + tmin_flare
			flare_mag_long = interp_y1(t_local_long)
			y_sim_long[mask_long] += flare_mag_long-max(flare_mag_long)

		mask_long2 = (t_base_long2 > t0 - tmin_flare) & (t_base_long2 < t0 + tmax_flare)
		if np.sum(mask_long2)!=0:
			t_local_long2 = t_base_long2[mask_long2] - t_base_long2[mask_long2].min() + tmin_flare
			flare_mag_long2 = interp_y1(t_local_long2)
			y_sim_long[mask_long2] += flare_mag_long2-max(flare_mag_long2)



	plt.figure()
	plt.scatter(t_base, y_sim)
	plt.scatter(t_base_long, y_sim_long)
	plt.scatter(t_base_long2, y_sim_long2)
	# plt.scatter(t_regular, y_sim_regular)
	plt.gca().invert_yaxis()
	plt.xlabel('Time (days)')
	plt.ylabel('Magnitude')
	plt.xlim(700, 800)
	plt.savefig(output_plot+ tic_id+'_'+str(i)+'.png')

			
	
	all_lc[ID]['%s_band_t_base_Roman_short'%band] = t_base
	all_lc[ID]['%s_band_t_base_Roman_long'%band] = t_base_long
	all_lc[ID]['%s_band_t_base_Roman_long2'%band] = t_base_long2
	all_lc[ID]['%s_band_Roman_m_official_sampling_short'%band] = y_sim
	all_lc[ID]['%s_band_Roman_m_official_sampling_long'%band] = y_sim_long
	all_lc[ID]['%s_band_Roman_m_official_sampling_long2'%band] = y_sim_long2
	all_lc[ID]['%s_band_Roman_m_regular_sampling'%band] = y_sim_regular
	all_lc[ID]['%s_band_Roman_time_regular_sampling'%band] = t_regular

	counter += 1


	if (counter/batch_size)%1==0:

		batch_number = int(counter/batch_size)
		pkl.dump(all_lc,
				 open(output_file+'FL_Roman_lc_batch_%i_out_of_%i_lcs_includes_regular.pkl'%(batch_number, num_lc),
		 	     'wb'))
		all_lc = {}

	if counter == num_lc:
		break



			
	
