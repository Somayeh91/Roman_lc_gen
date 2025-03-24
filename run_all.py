import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.optimize import minimize
from astropy.timeseries import LombScargle

from george import kernels
import george
from astropy.io import fits
from functions import *
from gp_model_params import info
from glob import glob
from tqdm import tqdm
import pickle as pkl
import argparse


from pathlib import Path
home = str(Path.home())

import warnings
# Suppress specific runtime warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def parse_options():
	"""Function to handle options speficied at command line
	"""
	parser = argparse.ArgumentParser(description='Process input parameters.')
	parser.add_argument('-type', action='store', default='T2CEP',
						help='What Category Do You Want to Fit?')
	parser.add_argument('-bands', action='store', default='I',
						help='What band are you fitting?')
	parser.add_argument('-obs', action='store', default='OGLE',
						help='What Observatory Does the Data is Coming From?')
	parser.add_argument('-directory', action='store',
						default='/Research_data/Bulge_Variables/',
						help='Specify the directory where the light curves are stored.')
	parser.add_argument('-output_directory', action='store',
						default='output/',
						help='Specify the output directory.')
	parser.add_argument('-number_lc', action='store',
						default=100,
						help='Specify the number of successgully fitted light curves models you request.')
	parser.add_argument('-lc_single_ID', action='store',
						default=np.nan,
						help='Specify the ID of a single light curve you want to fit')
	parser.add_argument('-add_regular_lc', action='store',
						default=False,
						help='Add gp fit predicted on a regular time sampling.')
	parser.add_argument('-all_bands', action='store',
						default=False,
						help='Fit data in all bands where a good fit can be achieved.')
	parser.add_argument('-save_output', action='store',
						default=True,
						help='Do you want to save the output file?')
	parser.add_argument('-plot_lc', action='store',
						default=True,
						help='Do you want to plot the result and save it?')

	# Parses through the arguments and saves them within the keyword args
	arguments = parser.parse_args()
	return arguments


args = parse_options()
tp = args.type
n_phs_original = info[tp]['n_phs']
direc_lcs = args.directory
output_direc = args.output_directory
number_lc = int(args.number_lc)
lc_single_ID = args.lc_single_ID
add_regular_lc = args.add_regular_lc
all_bands = args.all_bands
bands = args.bands.split(',')
save_output = args.save_output
plot_lc = args.plot_lc

direc = home + direc_lcs + '%s/'%tp
direc_list = np.sort(glob(direc+'*.fits'))
list_periods = np.genfromtxt(home + direc_lcs + '%s/BLG_%s_OGLE_periods.txt'%(tp,tp)
							, dtype='str')


lists = direc_list #[71:72]
all_templates = {}
counter = 0

# Read time sampling of Roman Galactic Bulge Surevy. Set the start time to zero.
Roman_sampling = np.loadtxt('lc_example/ulwdc1_208_W149.txt', usecols=0)
Roman_sampling = Roman_sampling - min(Roman_sampling)



for j, path in tqdm(enumerate(lists)):

	flags = np.zeros((len(bands)))
	metrics_all_bands = np.zeros((len(bands), 6))
	

	
	
	
	
	ID = path.split('/')[-1].split('_')[0]
	


	if isinstance(lc_single_ID, str):
		if not ID == lc_single_ID:
			continue

	
	print(ID)
	period = float(list_periods[list_periods[:, 0] == ID][0][1])
	
	period, df_ = read_fits(path, period=period,
						   tp = tp)
	
	if np.isnan(period):
		pass

	all_templates[ID] = {}

	for i, band in enumerate(bands):
		print('Reading %s band'%band)

		if all_bands:
			if i>0 and np.sum(flags) != 0:
				flags[i] = 1
				print('Error: All_bands is on. %s was removed because one band is missing.'%ID)
				continue
		try:
			df = df_[df_.band==band]
		except AttributeError:
			flags[i] = 1
			print('Error: %s band does not exit. Skipping...'%band)
			continue

		if len(df)<10:
			flags[i] = 1 
			print('Error: Found less than 10 datapoints. Skipping...')
			continue

		df = df.reset_index(drop=True)

		df_modified = prep_for_fit(df, period, info[tp])

		(df_roman, 
		   time_sampling_regular_final,
		   gp_y_regular_final,
		   gp_std_regular_final,
		   regular_sampling_fit,
		   gp_y_regular_fit,
		   gp_std_regular_fit,
		   gp_y_binned_fit, 
		   y_median,
		   metrics,
		   data) = run_all(df_modified, tp, period, info[tp], verbose=False)

		
		if not np.isinf(n_phs_original):
			phases = [n_phs_original-3, n_phs_original-2, n_phs_original-1, n_phs_original+1, n_phs_original+2, n_phs_original+3]
			best_phs = evaluate_fit(df_modified, tp, period, phases, metrics, info[tp], verbose=True)
		else:
			phases =[n_phs_original]
			best_phs = np.inf
			if len(find_valid_rows(np.array([metrics]), 
                   threshold=info[tp]['metric_threshold'],
                   threshold_std=info[tp]['metric_threshold_std']),
				   level = 3)==0:
			flags[i] = 1
        	print('at least one metric value did not pass the threshold, removed light curve.')
			

		if np.isnan(best_phs):
			flags[i] = 1
			print('Error: Evaluation failed (no good phase!). Skipping...')
			continue
		elif np.isinf(best_phs):
			pass
		else:
			info[tp]['n_phs'] = best_phs
			(df_roman, 
			   time_sampling_regular_final,
			   gp_y_regular_final,
			   gp_std_regular_final,
			   regular_sampling_fit,
			   gp_y_regular_fit,
			   gp_std_regular_fit,
			   gp_y_binned_fit, 
			   y_median,
			   metrics,
			   data) = run_all(df_modified, tp, period, info[tp], verbose=False)

			if len(find_valid_rows(np.array([metrics]), 
								   threshold=info[tp]['metric_threshold'],
								   threshold_std=info[tp]['metric_threshold_std']))==0:
				flags[i] = 1
				print('Error: Evaluation best_phs fit failed. Skipping...')
				continue


		
		

		
		all_templates[ID]['%s_band_m_observation'%band] = df.m.values
		all_templates[ID]['%s_band_time_bservation'%band] = df.t.values
		all_templates[ID]['%s_band_Roman_m_official_sampling'%band] = (df_roman.m).values + y_median
		if add_regular_lc:
			all_templates[ID]['%s_band_Roman_m_regular_sampling'%band] = gp_y_regular_final + y_median
			all_templates[ID]['%s_band_Roman_time_regular_sampling'%band] = time_sampling_regular_final

		metrics_all_bands[i, :] = metrics
		




		info[tp]['n_phs'] = n_phs_original

	all_templates[ID]['period'] = period
	
	if all_bands:
		if np.sum(flags)!=0:
			del all_templates[ID]
		else:
			counter += 1
			if (counter/10)%1 == 0:
				print('Collected %i events!'%counter)
			
	else:
		if np.sum(flags)==len(flags):
			del all_templates[ID]
		else:
			counter += 1
			if (counter/10)%1 == 0:
				print('Collected %i events!'%counter)

	for i, band in enumerate(bands):
		if all_bands:
			if np.sum(flags)!=0:
				continue
		else:
			if flags[i] == 1:
				continue

		if plot_lc:
			fig = example_plot_output(all_templates[ID]['%s_band_Roman_time_regular_sampling'%band], 
									  all_templates[ID]['%s_band_Roman_m_regular_sampling'%band], 
									  data, 
									  y_median,
									  regular_sampling_fit,
			   						  gp_y_regular_fit, 
			   						  period,
			   						  metrics_all_bands[i, :])
			fig.savefig(output_direc+'plots/%s/%s_Roman_lc_%s_%s_band.png' %(tp, tp, ID, band))

	
	if counter == number_lc:
		break
		
		
IDs = list(all_templates.keys())
if save_output:
	if all_bands:
		filename = '%s_Roman_lc_%i_all_bands.pkl' %(tp, counter)
		if add_regular_lc:
			filename = '%s_Roman_lc_%i_all_bands_includes_regular.pkl' %(tp, counter)
	else:
		if add_regular_lc:
			filename = '%s_Roman_lc_%i_includes_regular.pkl' %(tp, counter)
		else:
			filename = '%s_Roman_lc_%i.pkl' %(tp, counter)

	pkl.dump(all_templates, 
			 open(output_direc+filename,
				  'wb'))

