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




from pathlib import Path
home = str(Path.home())
tp = 'DN'
direc = home+'/Research_data/Bulge_Variables/%s/'%tp



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
t_regular = np.linspace(0, max(Roman_sampling_short), 750000)


direc = home+'/Research_data/Bulge_Variables/%s/'%tp
output_plot = 'output/plots/%s/'%tp
output_file = 'output/'
direc_list = np.sort(glob(direc+'*.fits'))

with open(output_file + 'DN_peaks_bank.pkl', 'rb') as file:
	peaks_bank = pkl.load(file)

band = 'I'

batch_size = 100

period = np.nan

counter = 0
n_sim_lc_per_obs_lc = 10
max_n_peaks = 6
all_lc = {}

for key in list(peaks_bank.keys()):

    print('Reading light curve %s...'%key)
    band = 'I'
    path = direc + '%s_multiband_lc.fits'%(str(key))
    # Read the light curve data
    period, df_ = read_fits(path, period=period, tp=tp)
    try:
        df = df_[df_.band == band]
        # print(df_.groupby('band').size())
    except AttributeError:
        print('Error: %s band does not exist. Skipping...' % band)
    
    all_peaks = peaks_bank[key]
    n_all_peaks = len(all_peaks)
    if n_all_peaks == 0:
        print('%s has no peaks.'%str(key))
        continue

    # n_peaks is a list of numbers for number of peaks for each final simulated light curve
    if n_all_peaks >= 6:
        n_peaks = np.random.choice(np.linspace(1, max_n_peaks, max_n_peaks, dtype=int), size=10)
    else:
        n_peaks = np.random.choice(np.linspace(1, n_all_peaks, n_all_peaks, dtype=int), size=10)
    

    noise_amp = 0.001
    baseline_mag = np.median(df_.m)
    y_base = np.ones_like(t_base) * baseline_mag + np.random.normal(0, noise_amp, size=len(t_base))
    y_base_long = np.ones_like(t_base_long) * baseline_mag + np.random.normal(0, noise_amp, size=len(t_base_long))
    y_base_long2 = np.ones_like(t_base_long2) * baseline_mag + np.random.normal(0, noise_amp, size=len(t_base_long2))
    y_base_regular = np.ones_like(t_regular) * baseline_mag + np.random.normal(0, noise_amp, size=len(t_regular))
    
    y_sim = y_base.copy()
    y_sim_long = y_base_long.copy()
    y_sim_long2 = y_base_long2.copy()
    y_sim_regular = y_base_regular.copy()

    for i in range(n_sim_lc_per_obs_lc):

        ID = str(key)+'_ind'+str(i)+'_'+str(n_peaks[i])
        all_lc[ID] = {}

        rand_peak_times = np.random.choice(t_base, size=n_peaks[i])

        y_sim = y_base.copy()
        y_sim_long = y_base_long.copy()
        y_sim_long2 = y_base_long2.copy()
        y_sim_regular = y_base_regular.copy()

        peak_ind = np.random.choice(np.linspace(0, n_all_peaks-1, n_all_peaks, dtype=int), size=n_peaks[i])
        counter1 = 0
        counter2 = 0
        for j in range(n_peaks[i]):

            peak_dict = all_peaks[peak_ind[j]]
            gp = peak_dict["gp"]
            xfit = peak_dict["xfit"]
            x = peak_dict["x"]
            t_of_max_flux = peak_dict["x"][np.argmin(peak_dict["yfit"])]
            dt_left = t_of_max_flux - x.min()
            dt_right = x.max() - t_of_max_flux   
            interp_y1 = interp1d(peak_dict["x_new"], peak_dict["y_new"], kind='linear', bounds_error=False, fill_value=np.nan)
            
            t0 = rand_peak_times[j]
            mask = (t_base > t0 - dt_left) & (t_base < t0 + dt_right)
            mask_regular = (t_regular > t0 - dt_left) & (t_regular < t0 + dt_right)
            
            
            
            if np.sum(mask)!=0:
                t_local = t_base[mask] - t_base[mask].min() + dt_left
                peak_mag = interp_y1(t_local-t_local.min())
                y_sim[mask] += peak_mag-max(peak_mag)
            
            if np.sum(mask_regular) != 0:
                t_local_regular = t_regular[mask_regular] - t_regular[mask_regular].min() + dt_left
            
                peak_mag_regular = interp_y1(t_local_regular-t_local_regular.min())
                y_sim_regular[mask_regular] += peak_mag_regular-max(peak_mag_regular)
            
            mask_long = (t_base_long > t0 - dt_left) & (t_base_long < t0 + dt_right)
            if np.sum(mask_long)!=0:
                counter1 += 1
                t_local_long = t_base_long[mask_long] - t_base_long[mask_long].min() + dt_left
                peak_mag_long = interp_y1(t_local_long-t_local_long.min())
                y_sim_long[mask_long] += peak_mag_long-max(peak_mag_long)
        
            mask_long2 = (t_base_long2 > t0 - dt_left) & (t_base_long2 < t0 + dt_right)
            if np.sum(mask_long2)!=0:
                counter2 += 1
                t_local_long2 = t_base_long2[mask_long2] - t_base_long2[mask_long2].min() + dt_left
                peak_mag_long2 = interp_y1(t_local_long2 - t_local_long2.min())
                y_sim_long2[mask_long2] += peak_mag_long2-max(peak_mag_long2)
        
        plt.figure()
        plt.scatter(t_base, y_sim)
        plt.scatter(t_base_long, y_sim_long)
        plt.scatter(t_base_long2, y_sim_long2)
        # plt.scatter(t_regular, y_sim_regular)
        plt.gca().invert_yaxis()
        if counter1>0:
            plt.text(0.1, 0.1, 'n_peaks in long_cad_1 is %i'%counter1, transform=plt.gca().transAxes)
        if counter2>0:
            plt.text(0.1, 0.2, 'n_peaks in long_cad_2 is %i'%counter2, transform=plt.gca().transAxes)
        plt.xlabel('Time (days)')
        plt.ylabel('Magnitude')
        # plt.xlim(700, 800)
        plt.savefig(output_plot + ID +'.png')


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
                     open(output_file+'DN_Roman_lc_batch_%i_lcs_includes_regular.pkl'%(batch_number),
                     'wb'))
            all_lc = {}
	

    



			
	
