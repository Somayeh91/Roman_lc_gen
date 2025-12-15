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

Roman_sampling_short_max_time = 981.99


def read_fits(path, period=None, tp='RRLYR', verbose=False):
    """
    Read FITS file containing light curve data and organize it into a DataFrame.
    Works for OGLE/UKIRT light curves saved with fits format in the Roman Galactic Bulge Google Drive.
    
    Parameters
    ----------
    path : str
        Path to the FITS file.
    period : float or None, optional
        Known period of the object. If None, period is estimated from the data.
    tp : str, default='RRLYR'
        Object type. Used to determine period range if period is not provided.
        Options: 'DN', 'CV', 'FL', or other periodic types.
    verbose : bool, default=False
        If True, print informational messages.
        
    Returns
    -------
    tuple
        - period : float
            Period of the object (estimated or provided)
        - df_OGLE : pandas.DataFrame
            DataFrame containing the following columns:
            - t: time values (shifted so min(t)=0)
            - m: magnitude values
            - e: magnitude errors
            - phase: phase values (0-1)
            - band: photometric band (I, V, H0, H1, K0, K1)
            
    Notes
    -----
    - Tries to read multiple bands from the FITS file: LC_I, LC_V, LC_H0, LC_H1, LC_K0, LC_K1
    - If period is not provided and object type is periodic, uses Lomb-Scargle periodogram
      to estimate period within the range specified in the 'info' dictionary
    - Non-periodic objects (DN, CV, FL) are not phased
    - Time is shifted so that the minimum time is 0 for easier processing
    """
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
        if not tp in ['DN', 'CV', 'FL']:
            p_range = info[tp]['p_range']
            phase, period = phase_refine(t, m, e, p_range) 
            phase = T_0_fixer(t, m, period, phase)
        else:
            phase = t
    else: 
        if not np.isnan(period):
            phase = (t/period)%1
            phase = T_0_fixer(t, m, period, phase)
        else:
            phase = t
        
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
    """
    Estimate period and compute phases using Lomb-Scargle periodogram.
    
    Parameters
    ----------
    t : numpy.ndarray
        Time values.
    m : numpy.ndarray
        Magnitude values.
    e : numpy.ndarray
        Magnitude errors.
    p_range : tuple
        Period range to search (min_period, max_period).
        
    Returns
    -------
    tuple
        - phase : numpy.ndarray
            Phase values (0-1) computed with the estimated period
        - p : float
            Estimated period (days)
            
    Notes
    -----
    - Uses 5-term Lomb-Scargle periodogram for better period estimation
    - Searches 100 equally spaced periods within the expanded range
      (p_range[0] - p_range[0]/10, p_range[1] + p_range[1]/10)
    - Returns the period with maximum power in the periodogram
    """
    
    period = np.linspace(p_range[0]-(p_range[0]/10), p_range[1]+(p_range[1]/10), 100)
    
    ls = LombScargle(t, m, e, nterms = 5)

    f_steps = 1./period
    power_window = ls.power(f_steps)
    
    p = (period[np.argmax(power_window)])
    
    phase = (t/p)%1

    return phase, p


def add_last_point(df, p):
    """
    Add an artificial data point at the end of the time series for continuity.
    Needed when the light curve's baseline is very short and we want to repeat variability.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Light curve DataFrame with columns: t, m, e, phase, band.
    p : float
        Period of the light curve.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional row containing data interpolated
        from the last phase position to fill the gap to Roman_sampling_short_max_time.
        
    Notes
    -----
    - The added point is at time Roman_sampling_short_max_time + 0.99 days
    - Magnitude, error, phase, and band are taken from the closest phase point
    - This helps ensure continuity when extending light curves
    """
    last_point_t = float(Roman_sampling_short_max_time + .99)
    last_point_m = df.m[np.argmin(np.abs(((df.t/p)%1).values-((Roman_sampling_short_max_time + .99/p)%1)))]
    last_point_e = df.e[np.argmin(np.abs(((df.t/p)%1).values-((Roman_sampling_short_max_time + .99/p)%1)))]
    last_point_phase = df.phase[np.argmin(np.abs(((df.t/p)%1).values-((Roman_sampling_short_max_time + .99/p)%1)))]
    last_point_band = df.band[np.argmin(np.abs(((df.t/p)%1).values-((Roman_sampling_short_max_time + .99/p)%1)))]

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
    Prepare input data for light curve processing, handling both multi-phase and single-phase cases.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Light curve DataFrame with columns: t, m, e.
    tp : str
        Object type (unused in current implementation).
    p : float
        Period of the light curve.
    info_ : dict
        Configuration dictionary containing:
        - 'n_phs': Number of phases to use (float or np.inf)
    t_max : float
        Maximum time value for the output.
        
    Returns
    -------
    tuple
        - x : numpy.ndarray
            Time or phase values (sorted)
        - y : numpy.ndarray
            Magnitude values (sorted corresponding to x)
        - e : numpy.ndarray
            Error values (sorted corresponding to x)
            
    Notes
    -----
    - If n_phs is finite: creates multi-phase folded light curve using create_multi_phase_repeating_folded_lc()
    - If n_phs is infinite (np.inf): uses original time series directly
    - Output arrays are always sorted by time/phase
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
    """
    Preprocess light curve data for Gaussian Process fitting.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Light curve DataFrame with columns: t, m, e.
    period : float
        Period of the light curve.
    info_ : dict
        Configuration dictionary containing:
        - 'p0_period': Initial period parameter for GP
        - 'n_phs': Number of phases to use
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed DataFrame ready for GP fitting.
        
    Notes
    -----
    - Shifts time so that minimum time is 0
    - For periodic objects with known period:
        * Adds last point if needed to reach Roman_sampling_short_max_time
        * Reduces gaps in phase-folded data
        * Extends baseline if needed for infinite phase fitting
    - For non-periodic objects:
        * Compresses large time gaps (>50 days)
        * Handles multiple light curve segments
    """
    if np.min(df.t)>0:
        df.t = df.t - np.min(df.t)

    if not np.isnan(info_['p0_period']):
        if (np.max(df.t))<Roman_sampling_short_max_time:
            df = add_last_point(df, period)
        df_modified = gap_reducer(df, period, info_)
        if (np.max(df_modified.t))<Roman_sampling_short_max_time and np.isinf(info_['n_phs']):
            df_modified = extend_baseline(df_modified, period)
        while np.max(df_modified.t.values)<Roman_sampling_short_max_time+period:
            len_0 = np.diff(df_modified.t.values)
            df_modified = gap_reducer(df_modified, period, info_)
            len_1 = np.diff(df_modified.t.values)
            if np.max(df_modified.t.values)>Roman_sampling_short_max_time:
                break
            if np.sum(len_0>50) == np.sum(len_1>50):
                break
    else:
        t_min = np.min(np.diff(df.t.values))
        inds = np.where(np.diff(df.t.values)>50)[0]
        for i, ind in enumerate(inds):
            time_gap = df.t[ind+1]-df.t[ind]+t_min
            df.loc[df.index[ind+1]:, 't'] -= time_gap
        n_lc = int(max(df.t)//Roman_sampling_short_max_time)
        df_modified = df
    return df_modified


def best_window_irregular(t, y, width=981.0):
    """
    Find the time window of specified width with the highest mean value.
    
    Parameters
    ----------
    t : numpy.ndarray
        1D array of times in days (monotonically increasing).
    y : numpy.ndarray
        1D array of values corresponding to times t.
    width : float, default=981.0
        Window width in days.
        
    Returns
    -------
    tuple
        - best_i : int
            Start index of best window
        - best_j : int
            End index of best window (inclusive)
        - start_time : float
            Start time of best window
        - end_time : float
            End time of best window
        - best_mean : float
            Mean value within best window
            
    Notes
    -----
    - Searches for window t[i:j] where t[j] <= t[i] + width
    - Returns the window with maximum mean(y[i:j])
    - Uses a tolerance of 5 days when finding j for each i
    """
    t = np.asarray(t)
    y = np.asarray(y)

    best_mean = -np.inf
    best_i = best_j = 0

    # for each start i, find the largest j with t[j] <= t[i] + width
    for i in range(len(t)):
        t_max = t[i]+width+5
        if t_max>max(t)-1:
            return best_i, best_j, float(t[best_i]), float(t[best_j]), float(best_mean)
        else:
            j = np.argmin(np.abs(t-t_max))+1
            m = np.mean(y[i:j])
            if m > best_mean:
                best_mean = m
                best_i, best_j = i, j

    return best_i, best_j, float(t[best_i]), float(t[best_j]), float(best_mean)


def create_regular_final_time_from_regular_fit(regular_sampling_fit, 
                                               gp_y_regular_fit, 
                                               gp_std_regular_fit,
                                               info_, 
                                               period, 
                                               t_max):
    """
    Extend regularly sampled GP predictions to cover desired time range.
    
    Parameters
    ----------
    regular_sampling_fit : numpy.ndarray
        Time samples from GP fitting.
    gp_y_regular_fit : numpy.ndarray
        GP mean predictions at regular_sampling_fit.
    gp_std_regular_fit : numpy.ndarray
        GP standard deviation predictions at regular_sampling_fit.
    info_ : dict
        Configuration dictionary containing 'n_phs' (number of phases).
    period : float
        Period of the light curve (np.nan for non-periodic).
    t_max : float
        Maximum desired time value.
        
    Returns
    -------
    tuple
        - regular_sampling_final : numpy.ndarray
            Extended time array up to t_max
        - gp_y_regular_final : numpy.ndarray
            Extended GP mean predictions
        - gp_std_regular_final : numpy.ndarray
            Extended GP standard deviation predictions
            
    Notes
    -----
    - For periodic objects: repeats pattern to fill time range
    - For non-periodic objects: extends baseline or truncates as needed
    - Handles both finite and infinite n_phs cases
    """
    n = info_['n_phs']  # Number of phases
    if not np.isnan(period):
        total_periods = int(((t_max / period) - (t_max / period) % 1) + 1)  # Compute total required periods
    else:
        total_periods = np.nan
    
    # If the maximum value in regular_sampling_fit exceeds t_max, truncate the data
    if max(regular_sampling_fit) > t_max:
        if np.isnan(total_periods):
            start_idx, end_idx, start_day, end_day, best_mean = best_window_irregular(regular_sampling_fit, -1*gp_y_regular_fit, width=t_max)
            gp_y_regular_final = gp_y_regular_fit[start_idx: end_idx]
            gp_std_regular_final = gp_std_regular_fit[start_idx: end_idx]
            regular_sampling_final = regular_sampling_fit[start_idx: end_idx]
        else:
            closest_idx = np.argmin(np.abs(regular_sampling_fit - t_max))
            max_tmp = regular_sampling_fit[min(closest_idx + 1, len(regular_sampling_fit) - 1)]
            gp_y_regular_final = gp_y_regular_fit[regular_sampling_fit <= max_tmp]
            gp_std_regular_final = gp_std_regular_fit[regular_sampling_fit <= max_tmp]
            regular_sampling_final = regular_sampling_fit[regular_sampling_fit <= max_tmp]
    else:
        if np.isnan(total_periods):
            num_x_n = 2
        else:
            num_x_n = int(total_periods / n - total_periods / n % 1 + 1)  # Number of repetitions needed
        
        # Handle the case where n is infinite (extend only twice)
        if np.isinf(n):
            if np.isnan(total_periods):
                t_last_ph = max(regular_sampling_fit)
            else:
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


def stats_binning(x, y, e, bins=100):
    """
    Bin data using statistical binning.
    
    Parameters
    ----------
    x : numpy.ndarray
        Independent variable values.
    y : numpy.ndarray
        Dependent variable values.
    e : numpy.ndarray
        Error values.
    bins : int, default=100
        Number of bins.
        
    Returns
    -------
    tuple
        - bin_middles : numpy.ndarray
            Middle points of each bin
        - bin_means : numpy.ndarray
            Mean y value in each bin
        - bin_errors : numpy.ndarray
            Median error in each bin divided by 50
            
    Notes
    -----
    - Uses scipy.stats.binned_statistic for binning
    - Bin errors are computed as median of individual errors in each bin / 50
    - Useful for reducing computational cost of GP fitting
    """
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
    """
    Reduce large gaps in phase-folded light curve data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Light curve DataFrame with columns: t, m, e.
    p : float
        Period of the light curve.
    info_ : dict
        Configuration dictionary containing 'n_phs'.
    th : float, default=100
        Threshold for gap detection in days.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with reduced gaps and extended baseline.
        
    Notes
    -----
    - Identifies gaps larger than threshold
    - For periodic gaps (>1 period), shifts subsequent times to reduce gap
    - Extends data by concatenating a copy shifted by one period
    - For infinite phase fitting, truncates at 1714 days
    """
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

    df_gapped = df_gapped.reset_index().drop('index', axis =1)

    df_temp = df_gapped.copy()
    df_temp.t = df_temp.t + ((df_gapped.t[len(df_gapped)-1]/p)- (df.t[len(df_gapped)-1]/p)%1)*p

    df_gapped = pd.concat([df_gapped, df_temp])
    if np.isinf(info_['n_phs']):
        df_gapped = df_gapped[df_gapped.t < 1714]

    df_gapped = df_gapped.sort_values(by=['t'])
    df_gapped = df_gapped.reset_index().drop('index', axis=1)
    return df_gapped


def extend_baseline(df, p):
    """
    Extend baseline of light curve by repeating pattern.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Light curve DataFrame.
    p : float
        Period of the light curve.
        
    Returns
    -------
    pandas.DataFrame
        Extended DataFrame with repeated patterns to fill baseline.
        
    Notes
    -----
    - Extends baseline to 1714 days by repeating from phase-matched point
    - Useful for ensuring sufficient data for GP fitting
    """
    max_t = np.max(df.t)
    max_t_phase = max_t/p
    coeff_diff = ((1714-max_t)/p - ((1714-max_t)/p)%1) +1 
    ind = df.index[np.argmin(np.abs(((df.t/p)%1).values-((max_t/p)%1)))]
    df_extend = df[df.index>ind].copy()
    df_extended = df.copy()
    for p_i in range(int(coeff_diff/max_t_phase)):
        df_extend.t = df_extend.t + (max_t_phase + (p_i+1)*int(coeff_diff/max_t_phase))*p
        df_extended = pd.concat((df_extended, df_extend)).reset_index().drop('index', axis=1)
    return df_extended


def create_multi_phase_repeating_folded_lc(n_phases, period, df, t_max):
    """
    Create multi-phase folded light curve by repeating phase pattern.
    
    Parameters
    ----------
    n_phases : int
        Number of phases to repeat.
    period : float
        Period of the light curve.
    df : pandas.DataFrame
        Light curve DataFrame with columns: t, m, e.
    t_max : float
        Maximum time for output.
        
    Returns
    -------
    tuple
        - x : numpy.ndarray
            Time values in days (converted from phase)
        - y : numpy.ndarray
            Magnitude values
        - e : numpy.ndarray
            Error values
            
    Notes
    -----
    - Folds data using n_phases * period as folding period
    - Converts phase to days for GP fitting
    - Useful for modeling multi-periodic behavior or capturing shape variations
    """
    one_phase = (df.t.values/(n_phases*period))%1
    x = (one_phase)
    y = (df.m.values)
    e = (df.e.values)

    x, y, e = convert_phase_to_days(x, y, e, n_phases, period, t_max)
    return x, y, e


def convert_phase_to_days(phs_array, y_array, e_array, n_phases, period, t_max):
    """
    Convert phase array to days for multi-phase representation.
    
    Parameters
    ----------
    phs_array : numpy.ndarray
        Phase values (0-1).
    y_array : numpy.ndarray
        Magnitude values.
    e_array : numpy.ndarray
        Error values.
    n_phases : int
        Number of phases in multi-phase representation.
    period : float
        Period of the light curve.
    t_max : float
        Maximum time value.
        
    Returns
    -------
    tuple
        - converted_x : numpy.ndarray
            Time values in days
        - converted_y : numpy.ndarray
            Magnitude values (same as input)
        - converted_e : numpy.ndarray
            Error values (same as input)
            
    Notes
    -----
    - Multiplies phase by n_phases * period to convert to days
    - For n_phases*period > t_max, uses single repetition
    """
    n = n_phases
    total_periods = int(((t_max/(period)) - (t_max/(period))%1)+1)
    num_x_n = int(total_periods/n - total_periods/n%1 +1)
    
    converted_x = phs_array*period*n
    converted_y = y_array
    converted_e = e_array
    
    return converted_x, converted_y, converted_e


def der(xy):
    """
    Compute first derivative of a function.
    
    Parameters
    ----------
    xy : tuple or list
        (y_values, x_values) where y_values is the function values
        and x_values is the independent variable.
        
    Returns
    -------
    numpy.ndarray
        Array containing [derivative_values, midpoints]
        
    Notes
    -----
    - Uses central differences
    - Returns derivative at midpoints between original x values
    """
    xder, yder = xy[1], xy[0]
    return np.array([np.diff(yder) / np.diff(xder), xder[:-1] + np.diff(xder) * 0.5])


def smoothness_gen(x, y, gp):
    """
    Second derivative of the function gp.predict(y, x).
    
    Parameters
    ----------
    x : numpy.ndarray
        Input locations.
    y : numpy.ndarray
        Observed values.
    gp : george.GP
        Gaussian Process model.
        
    Returns
    -------
    float
        Smoothness measure (sum of absolute second derivatives).
        
    Notes
    -----
    - Computes second derivative of GP predictions
    - Sum of absolute second derivatives used as smoothness penalty
    """
    return np.nansum(np.abs(der(der([gp.predict(y, x)[0], x]))), axis=1)[0]


def nll(p, y, x, gp, s):
    """
    Negative log-likelihood function for GP optimization.
    
    Parameters
    ----------
    p : numpy.ndarray
        Kernel parameters.
    y : numpy.ndarray
        Observed values.
    x : numpy.ndarray
        Input locations.
    gp : george.GP
        Gaussian Process model.
    s : float
        Smoothness penalty strength.
        
    Returns
    -------
    float
        Negative log-likelihood with smoothness penalty.
        
    Notes
    -----
    - Combines GP log-likelihood with smoothness penalty
    - Penalizes non-smooth functions to avoid overfitting
    - Returns large value (1e25) for invalid parameters
    """
    gp.kernel.parameter_vector = p
    try:
        smoothness = smoothness_gen(x, y, gp)
        smoothness = smoothness if np.isfinite(smoothness) \
                                   and ~np.isnan(smoothness) else 1e25
    except np.linalg.LinAlgError:
        smoothness = 1e25

    ll = gp.log_likelihood(y, quiet=True)
    ll -= smoothness ** s

    return -ll if np.isfinite(ll) else 1e25


def opt_gp(p0, gp, x, y, s=1):
    """
    Optimize GP kernel parameters.
    
    Parameters
    ----------
    p0 : numpy.ndarray
        Initial kernel parameters.
    gp : george.GP
        Gaussian Process model.
    x : numpy.ndarray
        Input locations.
    y : numpy.ndarray
        Observed values.
    s : float, default=1
        Smoothness penalty strength.
        
    Returns
    -------
    george.GP
        GP model with optimized kernel parameters.
        
    Notes
    -----
    - Uses scipy.optimize.minimize to minimize negative log-likelihood
    - Optimizes only first two kernel parameters
    - Updates gp.kernel.parameter_vector with optimized values
    """
    results = op.minimize(nll, [p0[0], p0[1]],
                          args=(y, x, gp, s))
    gp.kernel.parameter_vector = results.x

    return gp


def T_0_fixer(t, m, p, phase):
    """
    Adjust phase zero point to phase 0.25.
    
    Parameters
    ----------
    t : numpy.ndarray
        Time values.
    m : numpy.ndarray
        Magnitude values.
    p : float
        Period.
    phase : numpy.ndarray
        Original phase values.
        
    Returns
    -------
    numpy.ndarray
        Phase values shifted so phase 0.25 corresponds to minimum magnitude.
        
    Notes
    -----
    - Finds time of minimum magnitude (T0)
    - Shifts phases so that phase 0.25 corresponds to T0
    - Ensures all phases remain in [0, 1) range
    """
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
    """
    Prepare Gaussian Process model with initial parameters.
    
    Parameters
    ----------
    info_ : dict
        Configuration dictionary containing:
        - 'kernel': GP kernel
        - 'p0': Initial kernel parameters
        - 'p0_period': Function to compute initial period parameter
        - 'n_phs': Number of phases
    period : float
        Period of the light curve.
    verbose : bool, default=False
        If True, print progress messages.
        
    Returns
    -------
    george.GP
        GP model with initialized kernel.
        
    Notes
    -----
    - Sets initial kernel parameters based on period and n_phs
    - Uses HODLRSolver for efficient computation
    """
    kernel = info_['kernel']
    p0 = info_['p0']
    if np.isinf(info_['n_phs']):
        if np.isnan(period):
            pass
        else:
            p0[0] = info_['p0_period'](period)
    else:
        p0[0] = info_['p0_period'](period)
    if verbose:
        print('Setting up the GP...')
    gp = george.GP(kernel, solver=george.HODLRSolver)
    gp.kernel.parameter_vector = p0
    if verbose:
        print('Successfully set up the GP.')
    return gp


def binning_data(x, y, e, info_, n_bins=np.nan, verbose=False):
    """
    Bin data for GP fitting if needed.
    
    Parameters
    ----------
    x : numpy.ndarray
        Time/phase values.
    y : numpy.ndarray
        Magnitude values.
    e : numpy.ndarray
        Error values.
    info_ : dict
        Configuration dictionary containing:
        - 'fit_binned': Whether to bin data
        - 'n_phs': Number of phases
        - 'count_per_bins': Target points per bin (if fit_binned)
    n_bins : float, default=np.nan
        Number of bins (if specified).
    verbose : bool, default=False
        If True, print progress messages.
        
    Returns
    -------
    tuple
        - x_fit : numpy.ndarray
            Binned/processed x values
        - y_fit : numpy.ndarray
            Binned/processed y values (mean-subtracted)
        - e_fit : numpy.ndarray
            Binned/processed e values
        - regular_sampling_fit : numpy.ndarray
            Regular sampling grid for predictions
            
    Notes
    -----
    - If fit_binned is True, uses statistical binning
    - Subtracts median from y values for numerical stability
    - Creates regular sampling grid based on n_phs
    """
    fit_binned = info_['fit_binned']
    y_median = np.nanmedian(y) 
    n_phases = info_['n_phs']

    if fit_binned:
        if verbose:
            print('Binning the data...')
        x_fit, y_fit, e_fit = stats_binning(x, y, e, bins=n_bins)
        y_fit = y_fit - y_median
    else:
        x_fit, y_fit, e_fit = x, y, e
        y_fit = y_fit - y_median
        
    # create regular sampling with appropriate density
    if np.isinf(info_['n_phs']):
        regular_sampling_fit = np.linspace(0, max(x), int(max(x)))
    else:
        regular_sampling_fit = np.linspace(0, max(x), int(info_['n_phs']*50))
        
    return x_fit[~np.isnan(y_fit)], y_fit[~np.isnan(y_fit)], e_fit[~np.isnan(y_fit)], regular_sampling_fit


def fit_gp(x_fit, y_fit, e_fit, info_, gp, verbose=False):
    """
    Fit Gaussian Process to data.
    
    Parameters
    ----------
    x_fit : numpy.ndarray
        Input locations.
    y_fit : numpy.ndarray
        Mean-subtracted magnitude values.
    e_fit : numpy.ndarray
        Error values.
    info_ : dict
        Configuration dictionary containing:
        - 'gp_opt': Whether to optimize GP parameters
        - 'gp_opt_s_param': Smoothness penalty strength
    gp : george.GP
        GP model.
    verbose : bool, default=False
        If True, print progress messages.
        
    Returns
    -------
    george.GP
        Fitted GP model.
        
    Notes
    -----
    - Pre-computes covariance matrix factorization
    - Optionally optimizes kernel parameters
    """
    p0 = gp.kernel.parameter_vector

    # Pre-compute the factorization of the matrix.
    gp.compute(x_fit, e_fit)
    
    if info_['gp_opt']:
        gp = opt_gp(p0, gp, x_fit, y_fit, s=info_['gp_opt_s_param'])
    
    if verbose:
        print('GP was successfully was computed.')
    
    return gp


def predict_gp(gp, 
               y_binned, 
               y_median, 
               regular_sampling_fit,
               Roman_sampling_short,
               Roman_sampling_long,
               info_, 
               period,
               t_max, 
               verbose=False):
    """
    Predict light curve for Roman mission sampling.
    
    Parameters
    ----------
    gp : george.GP
        Fitted GP model.
    y_binned : numpy.ndarray
        Binned mean-subtracted magnitude values.
    y_median : float
        Median magnitude (for adding back).
    regular_sampling_fit : numpy.ndarray
        Regular sampling grid.
    Roman_sampling_short : numpy.ndarray
        Roman short cadence sampling times.
    Roman_sampling_long : numpy.ndarray
        Roman long cadence sampling times.
    info_ : dict
        Configuration dictionary.
    period : float
        Period of the light curve.
    t_max : float
        Maximum time for predictions.
    verbose : bool, default=False
        If True, print progress messages.
        
    Returns
    -------
    tuple
        - df_Roman : dict
            Dictionary with Roman light curve predictions
        - regular_sampling_final : numpy.ndarray
            Extended regular sampling
        - gp_y_regular_final : numpy.ndarray
            GP predictions on extended sampling
        - gp_std_regular_final : numpy.ndarray
            GP uncertainties on extended sampling
        - gp_y_regular_fit : numpy.ndarray
            GP predictions on original regular sampling
        - gp_std_regular_fit : numpy.ndarray
            GP uncertainties on original regular sampling
            
    Notes
    -----
    - Extends predictions to cover Roman observation window
    - Finds best window for Roman observations
    - Predicts at Roman sampling times
    """
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

    start_idx, end_idx, start_day, end_day, best_mean = best_window_irregular(regular_sampling_final, -1*gp_y_regular_final, width=Roman_sampling_short_max_time)
    
    y_Roman_short, cov_roman = gp.predict(y_binned, Roman_sampling_short+start_day)
    y_Roman_long, cov_roman = gp.predict(y_binned, Roman_sampling_long+start_day)

    df_Roman = {'t_short': Roman_sampling_short,
                 't_long': Roman_sampling_long, 
                 'm_short': y_Roman_short,
                 'm_long': y_Roman_long}
    if verbose:
        print('Successfully created Roman lightcurve.')

    return df_Roman, regular_sampling_final, gp_y_regular_final, gp_std_regular_final, gp_y_regular_fit, gp_std_regular_fit


def evaluate_fit(df_modified, tp, period, phases, metrics, info_, verbose=False):
    """
    Evaluate GP fit quality across different phase counts.
    
    Parameters
    ----------
    df_modified : pandas.DataFrame
        Preprocessed light curve data.
    tp : str
        Object type.
    period : float
        Period of the light curve.
    phases : list or numpy.ndarray
        Phase counts to evaluate.
    metrics : numpy.ndarray
        Previous metrics (unused in current implementation).
    info_ : dict
        Configuration dictionary containing:
        - 'metric_threshold': Threshold for metric values
        - 'metric_threshold_std': Threshold for standard deviation metric
    verbose : bool, default=False
        If True, print evaluation progress.
        
    Returns
    -------
    float
        Best phase count that passes evaluation criteria, or np.nan if none pass.
        
    Notes
    -----
    - Tests multiple phase counts (n_phs values)
    - Computes metrics for each phase count
    - Selects best phase count based on validation criteria
    - Returns np.nan if no phase count passes thresholds
    """
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
        (df_roman, time_sampling_regular_final, gp_y_regular_final, gp_std_regular_final, regular_sampling_fit, gp_y_regular_fit, gp_std_regular_fit, gp_y_binned_fit, y_median, metrics, data) = run_all(df_modified, tp, period, info_, verbose=False)
    
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


def find_valid_rows(matrix, threshold=0.01, threshold_std=0.01, level=4):
    """
    Find rows in matrix that satisfy validation criteria.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        2D array with shape (n_rows, n_metrics).
    threshold : float, default=0.01
        Threshold for metrics 3-6 (columns 2-5 in 0-indexed).
    threshold_std : float, default=0.01
        Threshold for metric 2 (column 1 in 0-indexed).
    level : int, default=4
        Strictness level: 4 (all criteria) or 3 (relaxed).
        
    Returns
    -------
    numpy.ndarray
        Indices of rows that satisfy all criteria.
        
    Notes
    -----
    Criteria (for level=4):
    1. Metrics 3-6 (columns 2-5) < threshold
    2. Metric 2 (column 1) < threshold_std
    3. std(metrics 4-6) / mean(metrics 4-6) < 1 for each row
    4. Metric 1 (column 0) < 2
    """
    # Handle empty matrix case early
    if matrix.shape[0] == 0:
        return np.array([], dtype=int)

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
    
    return final_valid_rows


def run_all(df_modified, tp, period, info_, verbose=False):
    """
    Complete pipeline for generating Roman light curve predictions.
    
    Parameters
    ----------
    df_modified : pandas.DataFrame
        Preprocessed light curve data.
    tp : str
        Object type.
    period : float
        Period of the light curve.
    info_ : dict
        Configuration dictionary.
    verbose : bool, default=False
        If True, print progress messages.
        
    Returns
    -------
    tuple
        - df_roman : dict
            Roman light curve predictions
        - time_sampling_regular_final : numpy.ndarray
            Final regular time sampling
        - gp_y_regular_final : numpy.ndarray
            GP predictions on final sampling
        - gp_std_regular_final : numpy.ndarray
            GP uncertainties on final sampling
        - regular_sampling_fit : numpy.ndarray
            Original regular sampling
        - gp_y_regular_fit : numpy.ndarray
            GP predictions on original sampling
        - gp_std_regular_fit : numpy.ndarray
            GP uncertainties on original sampling
        - gp_y_binned_fit : numpy.ndarray
            GP predictions at binned data points
        - y_median : float
            Median magnitude
        - metrics : numpy.ndarray
            Fit quality metrics
        - data : dict
            Intermediate data products
            
    Notes
    -----
    - Complete workflow: data prep -> binning -> GP fitting -> prediction -> metrics
    - Automatically adjusts binning strategy based on data size
    """
    # Read Roman time sampling
    Roman_sampling_short = np.load('lc_example/roman_times_shortcadence.npy')
    Roman_sampling_long = np.load('lc_example/roman_times_longcadence.npy')
    Roman_sampling_short_min = min(Roman_sampling_short)
    Roman_sampling_short = Roman_sampling_short - Roman_sampling_short_min
    Roman_sampling_long_min = min(Roman_sampling_long)
    Roman_sampling_long = Roman_sampling_long - Roman_sampling_long_min
    Roman_max_short_time = max(Roman_sampling_short)
    max_time_regular = 2000  # One of the produced resampled light curves will have a baseline of 2000 days.
    
    gp = prep_gp(info_, period)
    # x, y, e are either the full lc or an interval of n phases of them conducted by repeating a folded phase n times
    x, y, e = prep_input(df_modified, tp, period, info_, Roman_max_short_time)

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

    x_binned = x_binned[~np.isnan(e_binned)]
    y_binned = y_binned[~np.isnan(e_binned)]
    e_binned = e_binned[~np.isnan(e_binned)]

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
                                      Roman_sampling_short,
                                      Roman_sampling_long, 
                                      info[tp], 
                                      period,
                                      max_time_regular
                                      )

    print('max of fitted gp is '+str(min(gp_y_regular_fit)))
    gp_y_binned_fit, cov_tmp = gp.predict(y_binned, x_binned)
    metrics = get_metrics(x_binned, gp_y_binned_fit, regular_sampling_fit, gp_y_regular_fit, y_binned)

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
    """
    Create example visualization of light curve fitting results.
    
    Parameters
    ----------
    roman_x : numpy.ndarray
        Roman sampling times.
    roman_y : numpy.ndarray
        Roman predicted magnitudes.
    data : dict
        Dictionary containing original and fitted data.
    y_median : float
        Median magnitude.
    x_fit_regular : numpy.ndarray
        Regular sampling times.
    y_fit_regular : numpy.ndarray
        GP predictions on regular sampling.
    period : float
        Period of the light curve.
    metrics : numpy.ndarray
        Fit quality metrics.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object with light curve visualizations.
        
    Notes
    -----
    - Creates 2x2 subplot for periodic objects, 1x2 for non-periodic
    - Shows: original vs fit (time domain), Roman predictions (time domain),
      original vs fit (phase domain), Roman predictions (phase domain)
    - Inverts y-axis for magnitude plots
    """
    if not np.isnan(period):
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
        axs[1,0].legend(loc='upper right')

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
    else:
        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(data['x_binned'], data['y_binned']+y_median,color='b', label='observation')
        axs[0].plot(x_fit_regular, y_fit_regular+y_median,color='orange', label = 'GP fit on regular sampling')
        axs[0].plot(data['x_binned'], data['gp_y_binned']+y_median,color='red', label = 'GP fit on observation sampling')
        axs[0].text(0.05, 
                      0.05,
                      'std_all=%.2f,'
                      ' l2_bin_reg=%.9f,'
                      ' l2_bin=%.9f,'%(metrics[0], 
                                           metrics[1], 
                                           metrics[2]), 
                      transform = axs[0].transAxes)
        axs[0].text(0.05, 
                      0.1,
                      'l2_bin_part1=%.9f,'
                      ' l2_bin_part2=%.9f'
                      ' l2_bin_part2=%.9f'%(metrics[3],
                                            metrics[4], 
                                            metrics[4]),
                      transform = axs[0].transAxes)

        axs[1].scatter(roman_x, roman_y,color='b', label='Roman simulated')
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        axs[0].invert_yaxis()
        axs[1].invert_yaxis()
        axs[0].set_ylabel('Magnitude')
        axs[0].set_xlabel('Time (days)')
        axs[1].set_xlabel('Time (days)')

    fig = plt.gcf()
    fig.set_size_inches(15.0,12.0)
    return fig


def get_metrics(x_binned, gp_y_binned_fit, regular_sampling_fit, gp_y_regular_fit, y_binned):
    """
    Compute metrics to evaluate GP fit quality.
    
    Parameters
    ----------
    x_binned : numpy.ndarray
        Binned time/phase values.
    gp_y_binned_fit : numpy.ndarray
        GP predictions at binned points.
    regular_sampling_fit : numpy.ndarray
        Regular sampling grid.
    gp_y_regular_fit : numpy.ndarray
        GP predictions on regular grid.
    y_binned : numpy.ndarray
        Binned observed values.
        
    Returns
    -------
    numpy.ndarray
        Array of 6 metrics:
        0. std(y_binned)/std(gp_y_binned_fit)
        1. L2 distance between binned and regular predictions
        2. Mean squared error between y_binned and gp_y_binned_fit
        3-5. MSE for three equal segments of the light curve
        
    Notes
    -----
    - Metrics assess fit quality, smoothness, and consistency
    - Used for automatic phase count selection
    """
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
    """
    Compute L2 distance between two curves on common overlapping domain.
    
    Parameters
    ----------
    x1 : numpy.ndarray
        x-values of first curve.
    y1 : numpy.ndarray
        y-values of first curve.
    x2 : numpy.ndarray
        x-values of second curve.
    y2 : numpy.ndarray
        y-values of second curve.
        
    Returns
    -------
    float
        L2 distance (integral of squared difference) over overlapping domain.
        
    Notes
    -----
    - Interpolates both curves to common grid
    - Computes sqrt(∫(y1 - y2)² dx) over overlapping region
    - Uses linear interpolation
    """
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