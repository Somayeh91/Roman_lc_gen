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
from astropy.table import Table
from astropy.io import fits
from scipy.optimize import curve_fit
import re
from tqdm import tqdm
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


path = home+'/Research_data/Bulge_Variables/Flares_TESS/'
output_plot = 'output/plots/FL/'
output_file = 'output/flares/'

all_paths = np.sort(glob(path+'*.fits'))

band = 'TESS'
period = np.nan

baseline_bank = {}
flare_bank = {}
info_copy = info[tp].copy()

t_base = Roman_sampling_short
t_base_long = Roman_sampling_long
t_base_long2 = Roman_sampling_long2
t_regular = np.linspace(0, max(Roman_sampling_short), 750000)

for path in tqdm(all_paths):
    tic_id = path.split('/')[-1].split('_')[0].split('c')[-1]
    print(tic_id)
    with fits.open(path) as hdul:
        h = hdul[1].header
        hd = hdul[1].data
    t = Table(hd)

    flares = []
    # find all flare numbers present in the header
    flare_nums = sorted({
        int(re.search(r"F(\d+)", k).group(1))
        for k in h.keys()
        if re.match(r"F\d+-EXT", k)
    })
    
    for n in flare_nums:
        flare = {
            "flare_id": n,
            "ext": h.get(f"F{n}-EXT"),
            "btjd_start": h.get(f"F{n}-BTJD-START"),
            "btjd_peak": h.get(f"F{n}-BTJD-PEAK"),
            "btjd_stop": h.get(f"F{n}-BTJD-STOP"),
            "mag_start": h.get(f"F{n}-MAG-START"),
            "mag_peak": h.get(f"F{n}-MAG-PEAK"),
            "mag_stop": h.get(f"F{n}-MAG-STOP"),
        }
        flares.append(flare)
    
    df_header = pd.DataFrame(flares)
    df_lc = t.to_pandas()
    df = df_lc[["BTJD", "mag", "mag_err"]].rename(columns={"BTJD": "t", "mag": "m", "mag_err": "e"})

    t = df.t.values
    m = df.m.values
    e = df.e.values

    gp, data = rot_var_gp_fit(t, m, e, n_bins = 1000)

    m_data, t_data, m_err = data[0], data[1], data[2]


    y_base_regular = gp.predict(m_data, t_regular, return_var=True)[0]

    interp_baseline = interp1d(t_regular, y_base_regular, kind='linear', bounds_error=False, fill_value=np.nan)
    y_base = interp_baseline(t_base)
    y_base_long = interp_baseline(t_base_long)
    y_base_long2 = interp_baseline(t_base_long2)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(t_data, m_data, s=10, alpha=0.5, label="Binned data", zorder=3)
    ax.plot(t_base, y_base, color="red", lw=1.5, label="GP mean")
    ax.set_xlim(0, 10)          # swap to (0, 1000) for the full extrapolation
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.set_title("GP fit to TIC ID=%s"%tic_id)
    plt.tight_layout()
    plt.savefig(output_plot+tic_id+'_gp_baseline.png')
    baseline_bank[tic_id] = {}
    baseline_bank[tic_id]['%s_band_Roman_m_official_sampling_short'%band] = y_base
    baseline_bank[tic_id]['%s_band_Roman_m_official_sampling_long'%band] = y_base_long
    baseline_bank[tic_id]['%s_band_Roman_m_official_sampling_long2'%band] = y_base_long2
    baseline_bank[tic_id]['%s_band_Roman_m_regular_sampling'%band] = y_base_regular
    baseline_bank[tic_id]['%s_band_Roman_time_regular_sampling'%band] = t_regular

    

    

    for f in df_header.flare_id.values:
        if df_header[df_header.flare_id==f].btjd_peak.values[0]>max(df.t.values):
            continue
        fl_start = df_header[df_header.flare_id == f]['btjd_start'].values[0]
        fl_stop = df_header[df_header.flare_id == f]['btjd_stop'].values[0]
        df_fl6 = df[(df.t<fl_stop) & (df.t>fl_start) ]
        if np.min(df_fl6.t) > 0:
            df_fl6.loc[:, 't'] = df_fl6['t'] - np.min(df_fl6['t'])
        x, y, e = prep_input(df_fl6, 
                             tp, 
                             period, 
                             info_copy, 
                             Roman_max_short_time)

        dic = flare_fit(x, y, e, info_copy, period)
        flare_bank[tic_id+'_'+str(f)] = dic
    
        # print(info[tp])
        # plt.figure()
        # plt.scatter(x, y)
        # # plt.plot(dic["xfit"]-min(dic["xfit"]), dic["yfit"])
        # plt.scatter(dic["x_new"], dic["y_new"])
        # plt.title('TIC ID = %s, flare id = %i '%(tic_id, f))
        # plt.gca().invert_yaxis()
        # plt.savefig(output_plot+ tic_id+'_'+str(f)+'.png')


pkl.dump(baseline_bank, 
         open(output_file+'flare_baseline_bank.pkl',
              'wb'))

pkl.dump(flare_bank, 
         open(output_file+'flare_gp_fit_bank.pkl',
              'wb'))


    


    
    
    

