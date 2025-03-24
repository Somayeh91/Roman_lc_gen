import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from george import kernels
import george
import scipy.optimize as op
from gp_model_params import *
from astropy.io import fits
from scipy import stats
import matplotlib.pyplot as plt



def read_fits (path, period=None, tp = 'RRLYR', verbose=False):
    bands = ['LC_I', 'LC_V', 'LC_H0','LC_H1', 'LC_K0', 'LC_K1']
    filedirec = path
    hdul = fits.open(filedirec)
    t = np.asarray([])
    m = np.asarray([])
    e = np.asarray([])
    band = []
    for b in bands:
        try:
            t = np.concatenate((t, hdul[b].data['HJD']), axis=0)
            m = np.concatenate((m, hdul[b].data['mag']), axis=0)
            e = np.concatenate((e, hdul[b].data['mag_error']), axis=0)
            band = band + [b.split('_')[1]]*len(hdul[b].data['HJD'])
        except:
            if verbose:
                print('No %s band was found.'%(b.split('_')[1]))
            pass    
    
    if len(t)==0:
        print('No data found. Exit!')
        return np.nan, np.nan
    if period is None:
        p_range = info[tp]['p_range']
        phase, period = phase_refine(t, m, e, p_range) 
        phase = T_0_fixer(t, m, period, phase)
    else: 
        phase = (t/period)%1
        phase = T_0_fixer(t, m, period, phase)
        
    if np.min(t)>0:
        t_prime = t-np.min(t)
    else:
        t_prime = t
    
    
    df_OGLE = pd.DataFrame({'t': t_prime, 
                            'm': m,
                            'e': e,
                            'phase': phase,
                             'band': band})
    
    return period, df_OGLE



def phase_refine(t, m, e, p_range):
    
    period = np.linspace(p_range[0]-(p_range[0]/10), p_range[1]+(p_range[1]/10), 100)
    
    ls = LombScargle(t, m, e, nterms = 5)

    f_steps = 1./period
    power_window = ls.power(f_steps)
    
    p = (period[np.argmax(power_window)])
    
    phase = (t/p)%1

    return phase, p

def add_last_point(df, p):
    last_point_t = float(1713.99)
    last_point_m = df.m[np.argmin(np.abs(((df.t/p)%1).values-((1713.99/p)%1)))]
    last_point_e = df.e[np.argmin(np.abs(((df.t/p)%1).values-((1713.99/p)%1)))]
    last_point_phase = df.phase[np.argmin(np.abs(((df.t/p)%1).values-((1713.99/p)%1)))]
    last_point_band = df.band[np.argmin(np.abs(((df.t/p)%1).values-((1713.99/p)%1)))]

    df = pd.concat((df, pd.DataFrame(data=np.asarray([float(last_point_t),
                                      float(last_point_m),
                                      float(last_point_e),
                                      float(last_point_phase),
                                      last_point_band]).reshape((1, 5)),
                                   columns=df.columns))).reset_index().drop('index', axis =1)

    df = df.astype({'t':'float',
                   'm':'float',
                   'e':'float',
                   'phase':'float'})
    return df

def prep_input(df, tp, p, info_, t_max):
    """
    Prepares input data for light curve processing, handling both multi-phase and single-phase cases.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing light curve data with at least the following columns:
        - 't': Time values.
        - 'm': Magnitude values.
        - 'e': Error values.
    tp : unused parameter
        This parameter is currently unused in the function.
    p : float
        Period of the light curve, used if multi-phase processing is applied.
    info_ : dict
        Dictionary containing metadata, including:
        - 'n_phs': Number of phases to use for multi-phase light curve processing.

    Returns:
    --------
    x : numpy.ndarray
        Sorted time or phase values.
    y : numpy.ndarray
        Sorted magnitude values.
    e : numpy.ndarray
        Sorted error values.

    Notes:
    ------
    - If `info_['n_phs']` is finite, the function folds the light curve using `create_multi_phase_repeating_folded_lc()`.
    - If `info_['n_phs']` is infinite (`np.isinf(n_phases)`), the function directly extracts time, magnitude, and error values.
    - The returned arrays are always sorted by time or phase.
    - `tp` is included as a parameter but is not used in the function.
    """

    
    n_phases = info_['n_phs']
    
    if not np.isinf(n_phases):

        x, y, e = create_multi_phase_repeating_folded_lc(n_phases, p, df, t_max)
        y = y[x.argsort()]
        e = e[x.argsort()]
        x = x[x.argsort()]

    else:
        x, y, e = df.t.values, df.m.values, df.e.values
        y = y[x.argsort()]
        e = e[x.argsort()]
        x = x[x.argsort()]
        
    return x, y, e

def prep_for_fit(df, period, info_):
    if np.min(df.t)>0:
        df.t = df.t - np.min(df.t)
    
    if (np.max(df.t))<1713:
        df = add_last_point(df, period)
    df_modified = gap_reducer(df, period, info_)
    if (np.max(df_modified.t))<1713 and np.isinf(info_['n_phs']):
        df_modified = extend_baseline(df_modified, period)
    while np.max(df_modified.t.values)<1713+period:
        len_0 = np.diff(df_modified.t.values)
        df_modified = gap_reducer(df_modified, period, info_)
        len_1 = np.diff(df_modified.t.values)
        if np.max(df_modified.t.values)>1713:
            break
        if np.sum(len_0>50) == np.sum(len_1>50):
            break
    return df_modified

def create_regular_final_time_from_regular_fit(regular_sampling_fit, 
                                               gp_y_regular_fit, 
                                               gp_std_regular_fit,
                                               info_, 
                                               period, 
                                               t_max):
    """
    Extends a regularly sampled time series to cover a specified time range based on periodicity.
    
    Parameters:
    regular_sampling_fit (numpy array): Initial time samples.
    gp_y_regular_fit (numpy array): Corresponding function values at the sampled times.
    info_ (dict): Dictionary containing 'n_phs', the number of true periods per phase.
    period (float): The period of the signal.
    t_max (float): The maximum desired time value.
    
    Returns:
    tuple: (regular_sampling_final, gp_y_regular_final), where both are numpy arrays
           containing the extended time samples and corresponding function values.
    """
    n = info_['n_phs']  # Number of phase points per period
    total_periods = int(((t_max / period) - (t_max / period) % 1) + 1)  # Compute total required periods
    
    
    # If the maximum value in regular_sampling_fit exceeds t_max, truncate the data
    if max(regular_sampling_fit) > t_max:
        closest_idx = np.argmin(np.abs(regular_sampling_fit - t_max))
        max_tmp = regular_sampling_fit[min(closest_idx + 1, len(regular_sampling_fit) - 1)]

        gp_y_regular_final = gp_y_regular_fit[regular_sampling_fit <= max_tmp]
        gp_std_regular_final = gp_std_regular_fit[regular_sampling_fit <= max_tmp]
        regular_sampling_final = regular_sampling_fit[regular_sampling_fit <= max_tmp]
    else:
        num_x_n = int(total_periods / n - total_periods / n % 1 + 1)  # Number of repetitions needed
        
        # Handle the case where n is infinite (extend only twice)
        if np.isinf(n):
            t_last_ph = ((np.max(regular_sampling_fit)/period) - (np.max(regular_sampling_fit)/period)%1)*period
            needed_len = (t_max-t_last_ph)
            extra_regular_sampling_fit = regular_sampling_fit[regular_sampling_fit<needed_len]+t_last_ph
            exra_gp_y_regular_fit = gp_y_regular_fit[regular_sampling_fit<needed_len]
            exra_gp_std_regular_fit = gp_y_regular_fit[regular_sampling_fit<needed_len]
            regular_sampling_final = []
            gp_y_regular_final = []
            gp_std_regular_final = []
            regular_sampling_final = list(regular_sampling_fit) +\
                                     list(extra_regular_sampling_fit[extra_regular_sampling_fit>np.max(regular_sampling_fit)])
            gp_y_regular_final = list(gp_y_regular_fit) +\
                                 list(exra_gp_y_regular_fit[extra_regular_sampling_fit>np.max(regular_sampling_fit)])
            gp_std_regular_final = list(gp_std_regular_fit) +\
                                 list(exra_gp_std_regular_fit[extra_regular_sampling_fit>np.max(regular_sampling_fit)])
        else:
            # Extend the sampled time and corresponding values over multiple periods
            regular_sampling_final = []
            gp_y_regular_final = []
            gp_std_regular_final = []
            for i in range(num_x_n):
                regular_sampling_final += list(regular_sampling_fit + (i * n * period))
                gp_y_regular_final += list(gp_y_regular_fit)
                gp_std_regular_final += list(gp_std_regular_fit)
        
        # Convert lists to numpy arrays
        regular_sampling_final = np.asarray(regular_sampling_final)
        gp_y_regular_final = np.asarray(gp_y_regular_final)
        gp_std_regular_final = np.asarray(gp_std_regular_final)
    
    return regular_sampling_final, gp_y_regular_final, gp_std_regular_final
    
def stats_binning(x, y, e, bins = 100):

    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y,\
                                                         statistic='mean', bins=bins)
    r = (bin_edges[1] - bin_edges[0])/2
    bin_middles = []
    bin_errors = []
    for k in range(len(bin_edges)-1):
        ind_bin = (x>bin_edges[k]) & (x<bin_edges[k+1])
        bin_middles.append(bin_edges[k] + r ) 
        bin_errors.append(np.median(e[ind_bin])/50)

    bin_middles = np.asarray(bin_middles)
    bin_errors = np.asarray(bin_errors)
    return bin_middles, bin_means, bin_errors

def gap_reducer(df, p, info_, th=100):
    t_diff = np.diff(df['t'].values)
    t_diff = np.concatenate((t_diff, [0]), axis=0)

    indxs = df[t_diff>th].index.values
    df_gapped = df.copy(deep=True)

    for ind in indxs:
        if ind+1 >= len(df_gapped):
            continue

        gap = ((df_gapped.t[ind+1]-df_gapped.t[ind])/p)

        if (gap<1):
            pass
        else:
            coeff_diff = (((df_gapped.t[ind+1] - df_gapped.t[ind])/p) - 
                         ((df_gapped.t[ind+1] -df_gapped.t[ind])/p)%1)
            df_gapped.loc[df_gapped.index>ind, 't'] = df_gapped.t[ind+1:] - coeff_diff*p
            # if np.max(df_gapped.t)<1713:




    df_gapped = df_gapped.reset_index().drop('index', axis =1)

    df_temp = df_gapped.copy()

    df_temp.t = df_temp.t + ((df_gapped.t[len(df_gapped)-1]/p)- (df.t[len(df_gapped)-1]/p)%1)*p

    df_gapped = pd.concat([df_gapped, df_temp]) #df.append(df_temp)
    if np.isinf(info_['n_phs']):
        df_gapped = df_gapped[df_gapped.t < 1714]

    df_gapped = df_gapped.sort_values(by=['t'])


    df_gapped = df_gapped.reset_index().drop('index', axis =1)
    return df_gapped


def extend_baseline(df, p):
    max_t = np.max(df.t)
    max_t_phase = max_t/p
    coeff_diff = ((1714-max_t)/p - ((1714-max_t)/p)%1) +1 
    ind = df.index[np.argmin(np.abs(((df.t/p)%1).values-((max_t/p)%1)))]
    df_extend = df[df.index>ind].copy()
    df_extended = df.copy()
    for p_i in range(int(coeff_diff/max_t_phase)):
        
        df_extend.t = df_extend.t + (max_t_phase + (p_i+1)*int(coeff_diff/max_t_phase))*p
        df_extended = pd.concat((df_extended, df_extend)).reset_index().drop('index', axis =1)
    return df_extended

def create_multi_phase_repeating_folded_lc(n_phases, period, df, t_max):
    n_chunks = 4
    points_per_chunk = 500
    # tmp_t = []
    # tmp_m = []
    # tmp_e = []

    one_phase = (df.t.values/(n_phases*period))%1
    x = (one_phase)
    y = (df.m.values)
    e = (df.e.values)

    # for i in range(n_chunks):
    #     if not i == n_chunks-1:
    #         df_new = df[i*points_per_chunk: (i+1)*points_per_chunk]
    #         tmp_t += ((df_new.t.values/(n_phases*period))%1).tolist()
    #         tmp_m += (df_new.m.values).tolist()
    #         tmp_e += (df_new.e.values).tolist()
    #     else:
    #         df_new = df[i*points_per_chunk:]
    #         tmp_t += ((df_new.t.values/(n_phases*period))%1).tolist()
    #         tmp_m += (df_new.m.values).tolist()
    #         tmp_e += (df_new.e.values).tolist()
        
    # x = np.asarray(tmp_t)
    # y = np.asarray(tmp_m)
    # e = np.asarray(tmp_e)

    x,  y, e = convert_phase_to_days(x, y, e, n_phases, period, t_max)
    return x,  y, e

def convert_phase_to_days(phs_array, y_array, e_array, n_phases, period, t_max):
    converted_x = []
    converted_y = []
    converted_e = []
    n = n_phases
    total_periods = int(((t_max/(period)) - (t_max/(period))%1)+1)
    num_x_n =  int(total_periods/n -total_periods/n%1 +1)
    # if n*period > t_max:
    converted_x = phs_array*period*n
    converted_y = y_array
    converted_e = e_array
    # else:
    #     for i in range(num_x_n):
    #         converted_x += list((phs_array*n + (i * n))*period)
    #         converted_y += list(y_array)
    #         converted_e += list(e_array)
    #     converted_x = np.asarray(converted_x)
    #     converted_y = np.asarray(converted_y)
    #     converted_e = np.asarray(converted_e)
    return converted_x, converted_y, converted_e


def der(xy):
    xder, yder = xy[1], xy[0]
    return np.array([np.diff(yder) / np.diff(xder), xder[:-1] + np.diff(xder) * 0.5])


def smoothness_gen(x, y, gp):
    return np.nansum(np.abs(der(der([gp.predict(y, x)[0], x]))), axis=1)[0]


def nll(p, y, x, gp, s):
    gp.kernel.parameter_vector = p
    try:
        smoothness = smoothness_gen(x, y, gp)
        smoothness = smoothness if np.isfinite(smoothness) \
                                   and ~np.isnan(smoothness) else 1e25
    except np.linalg.LinAlgError:
        smoothness = 1e25

    ll = gp.log_likelihood(y, quiet=True)  # - (smoothness) #np.sum((y - pred[inds]**2)) #
    ll -= smoothness ** s

    return -ll if np.isfinite(ll) else 1e25

def opt_gp(p0, gp, x, y, s =1):
    results = op.minimize(nll, [p0[0], p0[1]],
                          args=(y,
                                x, gp, s))
    gp.kernel.parameter_vector = results.x

    return gp#, nll(p0, y, x, gp, s)

def T_0_fixer(t, m, p, phase):
    T0 = t[np.argmin(m)]
    
    phase0 = (T0/(p))%1
    temp = phase0 - (0.25)
    phase = [i-temp for i in phase]
    for i in range(len(phase)):
        if phase[i]<0:
            phase[i]=phase[i]+1
        if phase[i]>1:
            phase[i]=phase[i]-1
    return np.asarray(phase)

def prep_gp(info_, period, verbose=False):
    kernel = info_['kernel']
    p0 = info_['p0']
    if np.isinf(info_['n_phs']):
        p0[0] = info_['p0_period'](period)
    else:
        p0[0] = info_['p0_period'](period)
    if verbose:
        print('Setting up the GP...')
    gp = george.GP(kernel, solver= george.HODLRSolver)
    gp.kernel.parameter_vector = p0
    if verbose:
        print('Successfully set up the GP.')
    return gp

def binning_data(x, y, e, info_, n_bins=np.nan, verbose=False):
    fit_binned = info_['fit_binned']
    y_median = np.nanmedian(y) 
    n_phases = info_['n_phs']

    if fit_binned:
        if verbose:
            print('Binning the data...')
        x_fit, y_fit, e_fit = stats_binning(x, y, e, bins = n_bins)
        y_fit = y_fit - y_median
        
    else:
        x_fit, y_fit, e_fit = x, y, e
        y_fit = y_fit - y_median
        
    # create regular sampling with 100 pts per phase
    if np.isinf(info_['n_phs']):
        regular_sampling_fit = np.linspace(0, max(x), int(max(x)))
    else:
        regular_sampling_fit = np.linspace(0, max(x), int(info_['n_phs']*50))
        
        
    return x_fit[~np.isnan(y_fit)], y_fit[~np.isnan(y_fit)], e_fit[~np.isnan(y_fit)], regular_sampling_fit
    
    
def fit_gp(x_fit, y_fit, e_fit, info_, gp, verbose=False):
    p0 = gp.kernel.parameter_vector

    # Pre-compute the factorization of the matrix.
    gp.compute(x_fit, 
               e_fit)
    
    if info_['gp_opt']:
        gp = opt_gp(p0, 
                    gp, 
                    x_fit, 
                    y_fit, 
                    s =info_['gp_opt_s_param'])
    else:
        nll_opt = None
    if verbose:
        print('GP was successfully was computed.')
    
    return gp

def predict_gp(gp, 
               y_binned, 
               y_median, 
               regular_sampling_fit,
               Roman_sampling,
               info_, 
               period,
               t_max, 
               verbose=False):

    gp_y_regular_fit, cov = gp.predict(y_binned, regular_sampling_fit)

    gp_std_regular_fit = np.sqrt(np.diag(cov))
              
    if verbose:
        print('GP was successfully was applied to the new time baseline.')


    regular_sampling_final, gp_y_regular_final, gp_std_regular_final = create_regular_final_time_from_regular_fit(regular_sampling_fit, 
                                                                                                                  gp_y_regular_fit, 
                                                                                                                  gp_std_regular_fit,
                                                                                                                  info_, 
                                                                                                                  period, 
                                                                                                                  t_max)

    intpl = interp1d(regular_sampling_final, gp_y_regular_final)

    y_interpolated_Roman = intpl(Roman_sampling)
    df_Roman = pd.DataFrame({'t': Roman_sampling, 
                             'm': y_interpolated_Roman})
    if verbose:
        print('Successfully created Roman lightcurve.')

    return df_Roman, regular_sampling_final, gp_y_regular_final, gp_std_regular_final, gp_y_regular_fit, gp_std_regular_fit

    
def evaluate_fit(df_modified, tp, period, phases, metrics, info_, verbose=False):
    if verbose:
        print('Evaluating...')    
    metrics_tmp = np.zeros((len(phases),7))
    
    
    if len(find_valid_rows(np.array([metrics]), 
                           threshold=info_['metric_threshold'],
                           threshold_std=info_['metric_threshold_std']))==1:
        print('No need for evaluation. Passed.')
        return np.inf


    for i, phs in enumerate(phases):
        info[tp]['n_phs'] = phs
        if verbose:
            print('Testing number of phases = %i'%phs)
        (df_roman, time_sampling_regular_final, gp_y_regular_final, regular_sampling_fit, gp_y_regular_fit, gp_std_regular_fit, gp_y_binned_fit, y_median, metrics, data) = run_all(df_modified, tp, period, info_, verbose=False)
    
        metrics_tmp[i,0] = phs
        metrics_tmp[i,1:] = metrics
    
    info[tp]['n_phs'] = phases[0]+2
    
    selected_ind = find_valid_rows(metrics_tmp[:,1:], 
                                   threshold=info_['metric_threshold'],
                                   threshold_std=info_['metric_threshold_std'])

    if len(selected_ind) == 0:
        if verbose:
                print('at least one metric value did not pass the threshold, removed light curve.')
        return np.nan
    elif len(selected_ind) == 1:
        best_phs = metrics_tmp[:,0][selected_ind[0]]
        if verbose:
            print('Passed evaluation.')
            print('Metrics for the best phase = %i  are std_all=%.9f,'
                                                      ' l2_bin_reg=%.9f,'
                                                      ' l2_bin=%.9f,'
                                                      ' l2_bin_part1=%.9f,'
                                                      ' l2_bin_part2=%.9f'
                                                      ' l2_bin_part2=%.9f'%(best_phs,
                                                                           metrics_tmp[selected_ind[0], 1], 
                                                                           metrics_tmp[selected_ind[0], 2], 
                                                                           metrics_tmp[selected_ind[0], 3],
                                                                           metrics_tmp[selected_ind[0], 4],
                                                                           metrics_tmp[selected_ind[0], 5], 
                                                                           metrics_tmp[selected_ind[0], 6]))
        return best_phs
    else:
        final_ind = np.argmin(metrics_tmp[selected_ind,2])
        best_phs = metrics_tmp[selected_ind,0][final_ind]
        if verbose:
            print('Passed evaluation.')
            print('Metrics for the best phase = %i  are std_all=%.9f,'
                                                      ' l2_bin_reg=%.9f,'
                                                      ' l2_bin=%.9f,'
                                                      ' l2_bin_part1=%.9f,'
                                                      ' l2_bin_part2=%.9f'
                                                      ' l2_bin_part2=%.9f'%(best_phs,
                                                                           metrics_tmp[selected_ind,1][final_ind], 
                                                                           metrics_tmp[selected_ind,2][final_ind], 
                                                                           metrics_tmp[selected_ind,3][final_ind],
                                                                           metrics_tmp[selected_ind,4][final_ind],
                                                                           metrics_tmp[selected_ind,5][final_ind], 
                                                                           metrics_tmp[selected_ind,6][final_ind]))
        return best_phs
    

def find_valid_rows(matrix, threshold=0.01, threshold_std=0.01, level = 4):
    # Condition 1: All last four column values must be below the threshold
    valid_rows = np.all(matrix[:, 2:] < threshold, axis=1)

    # Condition 2: The second column values must be below the threshold_std
    valid_rows2 = matrix[:, 1] < threshold_std
    
    # Condition 3: std([x1, x2, x3]) / mean([x1, x2, x3]) < 1 for the last three columns
    last_three_cols = matrix[:, -3:]  # Extract last three columns
    std_ratio = np.array([np.std(last_three_cols[i])/last_three_cols[i] for i in range(matrix.shape[0])])

    # Condition 4:Avoid division by zero issues
    std_ratio[np.isnan(std_ratio)] = np.inf  # Set NaN results to infinity (to exclude them)
    
    valid_std_rows = np.all(std_ratio < 1, axis=1)

    # check the std of the fit versus data
    valid_std_rows2 = matrix[:, 0]<2
    
    # Final valid rows that meet both conditions
    if level==4:
        final_valid_rows = np.where(valid_rows & valid_rows2 & valid_std_rows & valid_std_rows2)[0]
    elif level==3:
        final_valid_rows = np.where(valid_rows & valid_rows2 & valid_std_rows2)[0]
    
    return final_valid_rows  # Returns the indices of rows meeting both conditions

def run_all(df_modified, tp, period, info_, verbose=False):
    
    # Read Roman time sampling
    Roman_sampling = np.loadtxt('lc_example/ulwdc1_208_W149.txt', usecols=0)
    Roman_sampling = Roman_sampling - min(Roman_sampling)
    Roman_max_time = max(Roman_sampling)
    max_time_regular = 2000
    
    # Read and apply Roman noise function
    # noise_fun = noise_function('cycle6_snr_curve.txt')
    
    gp = prep_gp(info_, period)
    # x, y, e are either the full lc or an intrval of n phases of them conducted by repeating a folded phase n times
    x, y, e = prep_input(df_modified, tp, period, info_, Roman_max_time)


    if not np.isinf(info_['n_phs']):
        if len(x)/info_['n_phs']>150:
            if info_['fit_binned']==False:
                info_['fit_binned'] = True
            tot_ideal_len = info_['n_phs']*info_['count_per_bins']
            x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, 
                                                                              y, 
                                                                              e, 
                                                                              info_, 
                                                                              n_bins=tot_ideal_len)
        else:
            info_['fit_binned'] = False
            x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, 
                                                                              y, 
                                                                              e, 
                                                                              info_, 
                                                                              n_bins=np.nan)
    else:
        if len(x)>1500:
            info_['fit_binned'] = True
            tot_ideal_len = 1500
            x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, 
                                                                              y, 
                                                                              e, 
                                                                              info_, 
                                                                              n_bins=tot_ideal_len)
            info_['fit_binned'] = False
        else:
            x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, 
                                                                          y, 
                                                                          e, 
                                                                          info_, 
                                                                          n_bins=np.nan)
    
        

    gp = fit_gp(x_binned, y_binned, e_binned, info_, gp)
    
    


    y_median = np.median(y)
    
    (df_roman, 
     time_sampling_regular_final,
     gp_y_regular_final,
     gp_std_regular_final,
     gp_y_regular_fit,
     gp_std_regular_fit) = predict_gp(gp, 
                                        y_binned, 
                                        y_median, 
                                        regular_sampling_fit, 
                                        Roman_sampling, 
                                        info[tp], 
                                        period,
                                        max_time_regular
                                        )

    gp_y_binned_fit, cov_tmp = gp.predict(y_binned, x_binned)
    metrics = get_metrics(x_binned, gp_y_binned_fit, regular_sampling_fit, gp_y_regular_fit, y_binned) #np.nansum((y_binned-(gp_y_binned_fit))**2)/len(y_binned)

    data = {}
    data['y'] = y
    data['x'] = x
    data['e'] = e
    data['y_binned'] = y_binned
    data['x_binned'] = x_binned
    data['e_binned'] = e_binned
    data['gp_y_binned'] = gp_y_binned_fit
    data['gp'] = gp

    return (df_roman, 
           time_sampling_regular_final,
           gp_y_regular_final,
           gp_std_regular_final,
           regular_sampling_fit,
           gp_y_regular_fit,
           gp_std_regular_fit,
           gp_y_binned_fit, 
           y_median,
           metrics,
           data)
    

def example_plot_output(roman_x, roman_y, data, y_median, x_fit_regular, y_fit_regular, period, metrics):
    fig, axs = plt.subplots(2, 2)




    axs[0,0].scatter(data['x_binned'], data['y_binned']+y_median,color='b', label='observation')
    axs[0,0].plot(x_fit_regular, y_fit_regular+y_median,color='orange', label = 'GP fit on regular sampling')
    axs[0,0].plot(data['x_binned'], data['gp_y_binned']+y_median,color='red', label = 'GP fit on observation sampling')
    axs[0,0].text(0.05, 
                  0.05,
                  'std_all=%.2f,'
                  ' l2_bin_reg=%.9f,'
                  ' l2_bin=%.9f,'%(metrics[0], 
                                       metrics[1], 
                                       metrics[2]), 
                  transform = axs[0,0].transAxes)
    axs[0,0].text(0.05, 
                  0.1,
                  'l2_bin_part1=%.9f,'
                  ' l2_bin_part2=%.9f'
                  ' l2_bin_part2=%.9f'%(metrics[3],
                                        metrics[4], 
                                        metrics[4]),
                  transform = axs[0,0].transAxes)

    axs[1,0].scatter(roman_x, roman_y,color='b', label='Roman simulated')

    axs[0,1].scatter((data['x_binned']/period)%1, data['y_binned']+y_median,color='b', label='observation')
    axs[0,1].plot((x_fit_regular/period)%1, y_fit_regular+y_median,color='orange', marker='o', linestyle='', markersize=3, label = 'GP fit on regular sampling')
    axs[0,1].plot((data['x_binned']/period)%1, data['gp_y_binned']+y_median,color='red', marker='o', linestyle='', markersize=3, label = 'GP fit on observation sampling')

    axs[1,1].scatter((roman_x/period)%1, roman_y,color='b', label='Roman simulated')
    axs[1,1].plot((x_fit_regular/period)%1, y_fit_regular+y_median,color='orange', marker='o', linestyle='', markersize=3, label = 'GP fit on regular sampling')

    axs[0,0].legend(loc='upper right')
    axs[0,1].legend(loc='upper right')
    axs[1,1].legend(loc='upper right')
    axs[1,0].legend(loc='upper right'
        )





    axs[0,0].invert_yaxis()
    axs[0,1].invert_yaxis()
    axs[1,1].invert_yaxis()
    axs[1,0].invert_yaxis()

    axs[0,0].set_ylabel('Magnitude')
    axs[1,0].set_ylabel('Magnitude')

    axs[1,0].set_xlabel('Time (days)')
    axs[1,1].set_xlabel('Phase')

    axs[0,0].set_title('Full light curves')
    axs[0,1].set_title('Phase-folded light curves')




    fig = plt.gcf()
    fig.set_size_inches(15.0,12.0)
    return fig

def get_metrics(x_binned, gp_y_binned_fit, regular_sampling_fit, gp_y_regular_fit, y_binned):
    all_lc_metric = np.nansum((y_binned-(gp_y_binned_fit))**2)/len(y_binned)

    lc_part1_metric = np.nanmean((y_binned[:int(len(y_binned)/3)]-
                      (gp_y_binned_fit[:int(len(y_binned)/3)]))**2)
    lc_part2_metric = np.nanmean((y_binned[int(len(y_binned)/3):2*int(len(y_binned)/3)]-
                          (gp_y_binned_fit[int(len(y_binned)/3):2*int(len(y_binned)/3)]))**2)
    lc_part3_metric = np.nanmean((y_binned[2*int(len(y_binned)/3):int(len(y_binned))]-
                          (gp_y_binned_fit[2*int(len(y_binned)/3):int(len(y_binned))]))**2)
    lc_all_metric2 = l2_distance(x_binned, gp_y_binned_fit, regular_sampling_fit, gp_y_regular_fit)

    
    all_lc_std_metric = np.std(y_binned)/np.std(gp_y_binned_fit)
    return np.array([all_lc_std_metric, lc_all_metric2, all_lc_metric, lc_part1_metric, lc_part2_metric, lc_part3_metric])


def l2_distance(x1, y1, x2, y2):
    # Find the overlapping region
    x_min = max(min(x1), min(x2))
    x_max = min(max(x1), max(x2))

    # Define a common time grid only within the overlap range
    common_x = np.linspace(x_min, x_max, num=500)  # Use a dense grid for accuracy

    # Interpolate y1 and y2 within the common range
    interp_y1 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value=np.nan)
    interp_y2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value=np.nan)

    # Evaluate the interpolated values
    y1_interp = interp_y1(common_x)
    y2_interp = interp_y2(common_x)

    # Remove NaN values that may occur due to boundary issues
    valid_mask = ~np.isnan(y1_interp) & ~np.isnan(y2_interp)
    y1_interp = y1_interp[valid_mask]
    y2_interp = y2_interp[valid_mask]

    # Compute the L2 distance
    l2_dist = np.sqrt(np.sum((y1_interp - y2_interp) ** 2) * (common_x[1] - common_x[0]))

    return l2_dist
