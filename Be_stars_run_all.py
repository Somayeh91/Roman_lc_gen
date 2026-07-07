import pickle as pkl
import re
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from astropy.io import fits
from astropy.table import Table
import george
from george import kernels
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyArrowPatch
from functions import (
	read_fits, prep_input, prep_gp, fit_gp,
	binning_data, multiscale_peaks, predict_gp,
	get_metrics, l2_distance, find_valid_rows, generate_points
)
from gp_model_params import info
from tqdm.notebook import tqdm
import time
from scipy.interpolate import CubicSpline




# Global matplotlib style for paper figures
plt.rcParams.update({
	"font.family"      : "serif",
	"font.size"        : 13,
	"axes.labelsize"   : 14,
	"axes.titlesize"   : 14,
	"legend.fontsize"  : 11,
	"xtick.direction"  : "in",
	"ytick.direction"  : "in",
	"xtick.minor.visible": True,
	"ytick.minor.visible": True,
	"figure.dpi"       : 120,
})

# Colour palette (colour-blind friendly)
C0, C1, C2, C3, C4 = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"


# Paths and source-type configuration
home = str(Path.home())

TP   = 'Be'
BAND = 'G'        # Gaia G-band

DATA_PATH   = home + '/Research_data/GAIA BE stars/Gaia_Be_stars_candidates/gaia_science_alerts_be_stars_lightcurves/'
OUTPUT_PLOT = f'output/plots/{TP}/'
OUTPUT_FILE = 'output/'

fit_plot = True
Path(OUTPUT_PLOT).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FILE).mkdir(parents=True, exist_ok=True)

# Be stars are non-periodic — period is set to NaN throughout
period = np.nan

# Number of resampled light curves per each GAIA light curve
n_lc = 5

lc_except_list = ['Gaia19cef', 'Gaia19cex', 'Gaia19cug', 'Gaia19czw', 'Gaia19czy',
       'Gaia19ews', 'Gaia20ayv', 'Gaia20cfq', 'Gaia20dff', 'Gaia20fcq',
       'Gaia21ahr', 'Gaia21cqi', 'Gaia21dck', 'Gaia21edd', 'Gaia22avg',
       'Gaia22ayd', 'Gaia22bra', 'Gaia22bxd', 'Gaia22duz', 'Gaia22dyq',
       'Gaia22eaz', 'Gaia22enp', 'Gaia23ala', 'Gaia23bef', 'Gaia23bgy',
       'Gaia23biz', 'Gaia23bsh', 'Gaia23cve']

# Load pre-computed Roman Galactic Bulge cadence time arrays
Roman_sampling_short  = np.load('lc_example/roman_times_shortcadence.npy')
Roman_sampling_long   = np.load('lc_example/roman_times_longcadence.npy')
Roman_sampling_long2  = np.load('lc_example/roman_times_longcadence2.npy')


Roman_sampling_long   -= Roman_sampling_long.min()
Roman_sampling_long2  -= Roman_sampling_short.min()
Roman_sampling_short  -= Roman_sampling_short.min()

Roman_max_short_time = Roman_sampling_short.max()

t_base       = Roman_sampling_short
t_base_long  = Roman_sampling_long
t_base_long2 = Roman_sampling_long2

# Dense regular grid for smooth curve evaluation
t_regular = np.linspace(0, Roman_max_short_time, 750_000)

print(f"Short cadence  : {len(t_base):,} pts  |  span = {Roman_max_short_time:.1f} d")
print(f"Long cadence 1 : {len(t_base_long):,} pts  |  span = {t_base_long.max():.1f} d")
print(f"Long cadence 2 : {len(t_base_long2):,} pts  |  span = {t_base_long2.max():.1f} d")

# Discover all available light curve CSVs
lc_list = sorted(glob(DATA_PATH + '*.csv'))
print(f"Found {len(lc_list)} light curve files.")

band = 'G'
all_lc = {}
all_lc['time_sampling'] = {}

all_lc['time_sampling']['%s_band_t_base_Roman_short'%band] = t_base
all_lc['time_sampling']['%s_band_t_base_Roman_long'%band] = t_base_long
all_lc['time_sampling']['%s_band_t_base_Roman_long2'%band] = t_base_long2
all_lc['time_sampling']['%s_band_Roman_time_regular_sampling'%band] = t_regular

all_counter = 0
for d, demo_path in tqdm(enumerate(lc_list)):


	demo_id   = Path(demo_path).stem
	all_lc[demo_id]={}
	ID = demo_id
	print(d, ID)

	if ID == 'Gaia23cne':
		# Gap larger than 500 days
		continue
	if ID in lc_except_list:
		# GP run on all showed s_param=1 works better for these
		info[TP]['gp_opt_s_param'] = 1
	else:
		info[TP]['gp_opt_s_param'] = 1.5


	# Read CSV and rename columns to pipeline convention
	loaded_df = pd.read_csv(demo_path).reset_index(drop=True)
	df        = loaded_df.rename(columns={"JD(TCB)": "t", "mag": "m", "err": "e"})

	# Normalise time axis
	if df.t.min() > 0:
		df['t'] = df['t'] - df['t'].min()

	# Compress observing gaps > 500 days

	inds  = np.where(np.diff(df.t.values) > 500)[0]
	t_min = np.min(np.diff(df.t.values))

	df_modified = df.copy()
	for ind in inds:
		time_gap = df_modified.t.iloc[ind + 1] - df_modified.t.iloc[ind] + t_min
		df_modified.loc[df_modified.index[ind + 1]:, 't'] -= time_gap

	df_modified = df_modified.copy()
	scale_factor = np.max(df_modified['t'])/5
	df_modified['t'] = df_modified['t']/scale_factor

	# Initialise GP and build multi-phase input array
	gp = prep_gp(info[TP], period)

	x, y, e = prep_input(df_modified, TP, period, info[TP], Roman_max_short_time)



	# Bin the data to reduce GP matrix size

	if len(x) > 100:
		info[TP]['fit_binned'] = True
		x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(
			x, y, e, info[TP], n_bins=100
		)
	else:
		info[TP]['fit_binned'] = False
		x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(
			x, y, e, info[TP], n_bins=np.nan
		)

	# Drop bins whose error estimate is NaN (empty bins at the edges)
	valid    = ~np.isnan(e_binned)
	x_binned = x_binned[valid]
	y_binned = y_binned[valid]
	e_binned = e_binned[valid]

	y_median = np.percentile(y, 90)   # use 90th-percentile as baseline (faintest typical flux)

	# Fit GP to the binned data

	t0 = time.time()
	gp = fit_gp(x_binned, y_binned, e_binned, info[TP], gp)


	# GP mean evaluated at binned data points


	gp_y_binned_fit, cov_binned = gp.predict(y_binned, x_binned, return_cov=True)
	gp_std_binned               = np.sqrt(np.diag(cov_binned))

	# Build smooth display curve via spline (avoids dense-grid GP ringing)
	sort_idx      = np.argsort(x_binned)
	cs_fit        = CubicSpline(x_binned[sort_idx], gp_y_binned_fit[sort_idx])
	cs_std        = CubicSpline(x_binned[sort_idx], gp_std_binned[sort_idx])
	x_display     = np.linspace(x_binned.min(), x_binned.max(), 800)
	y_display     = cs_fit(x_display)
	std_display   = np.abs(cs_std(x_display))

	if fit_plot:
		fig, axes = plt.subplots(2, 1, figsize=(13, 7),
								 gridspec_kw={'height_ratios': [2.5, 1]})

		ax = axes[0]
		ax.scatter(x_binned * 500, y_binned + y_median,
				   s=18, alpha=0.8, color='grey', zorder=3, label='Binned Gaia data')
		ax.plot(x_display * 500, y_display + y_median,
				color=C1, lw=2.0, zorder=4, label='GP mean (splined)')
		ax.fill_between(x_display * 500,
						y_display + y_median - std_display,
						y_display + y_median + std_display,
						color=C1, alpha=0.2, label=r'GP $1\sigma$')
		ax.invert_yaxis()
		ax.set_ylabel('G magnitude')
		ax.set_title(f'GP fit to binned Gaia data — {demo_id}')
		ax.legend()
		ax.xaxis.set_minor_locator(AutoMinorLocator())

		# Residuals
		resid = y_binned - gp_y_binned_fit
		axes[1].scatter(x_binned * 500, resid, s=8, alpha=0.7, color='grey')
		axes[1].axhline(0, color=C1, lw=1.2, ls='--')
		axes[1].set_xlabel('Time (days)')
		axes[1].set_ylabel('Residual (mag)')
		axes[1].xaxis.set_minor_locator(AutoMinorLocator())

		ax.text(0.05, 0.05, np.sum((y_binned - gp_y_binned_fit)**2), size=20, transform=ax.transAxes)
		fig.savefig(OUTPUT_PLOT + demo_id + '_gp_fit.png')



	baseline_mag = max(y_binned + y_median)


	counter = 0

	x_binned_tmp = x_binned * scale_factor
	t_base_exc_edge = x_binned_tmp[x_binned_tmp < (max(x_binned_tmp)-max(t_base))]

	t_start_points = generate_points((max(x_binned_tmp)-max(t_base)), min_gap=300, max_gap=350)

	for i in range(len(t_start_points)):

		
		


		t_s = t_start_points[i]
		t_local = t_base + t_s
		outburst_mag = cs_fit(t_local/scale_factor)
		y_sim = outburst_mag
		ptp_init = np.ptp(y_sim)

		if ptp_init<0.1:
			# print('Generated lc was rejected due to low variability')
			continue

		t_local_long = t_base_long + t_s
		outburst_mag_long = cs_fit(t_local_long/scale_factor)
		y_sim_long = outburst_mag_long

		t_local_long2 = t_base_long2 + t_s
		outburst_mag_long2 = cs_fit(t_local_long2/scale_factor)
		y_sim_long2 = outburst_mag_long2

		t_local_regular = t_regular + t_s
		outburst_mag_regular = cs_fit(t_local_regular/scale_factor)
		y_sim_regular = outburst_mag_regular



		ID_i = 't_start_%i'%int(t_s)
		all_lc[ID][ID_i] = {}




		
		
		all_lc[ID][ID_i]['%s_band_Roman_m_official_sampling_short'%band] = y_sim
		all_lc[ID][ID_i]['%s_band_Roman_m_official_sampling_long'%band] = y_sim_long
		all_lc[ID][ID_i]['%s_band_Roman_m_official_sampling_long2'%band] = y_sim_long2
		all_lc[ID][ID_i]['%s_band_Roman_m_regular_sampling'%band] = y_sim_regular


		plt.figure()
		plt.scatter(t_base+t_s, y_sim+y_median,
					s=10, alpha=0.8, color='C0', zorder=3, label='Roman Resampled data')
		plt.plot(x_display * scale_factor, y_display+y_median,
				color=C1, lw=2.0, zorder=4, label='GP mean (splined)')
		plt.scatter(x_binned * scale_factor, y_binned+y_median,
				   s=18, alpha=0.8, color='grey', zorder=3, label='Binned Gaia data')
		# plt.axvline(t_rise)
		# plt.axvline(t_min)
		# plt.axvline(t_max)
		ax = plt.gca()
		ax.invert_yaxis()
		ax.legend()
		ax.set_xlabel('Time (days)')
		ax.set_ylabel('G magnitude')
		ax.xaxis.set_minor_locator(AutoMinorLocator())

		plt.savefig(OUTPUT_PLOT + str(demo_id)+'_'+str(counter)+'.png')
		counter += 1


	# print("Generafted %i light curves for %s."%(counter, demo_id))
	all_counter += counter

pkl.dump(all_lc, 
		 open(OUTPUT_FILE + '%s_Roman_lc_%i_all_bands_includes_regular.pkl' %(TP, all_counter),
			  'wb'))
