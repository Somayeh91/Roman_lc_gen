import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from george import kernels
import george
import scipy.optimize as op
from gp_model_params import info
from astropy.io import fits
from scipy import stats


def noise_function(noise_snr_file):
    
    noise_roman = np.loadtxt(str(noise_snr_file))
    
    noise_intpl = interp1d(noise_roman[:,0], noise_roman[:,1])
    
    return noise_intpl

def read_OGLE_lc (filename, p):
    filedirec = 'lc_example/' + filename + '.dat'
    t, m, e = np.loadtxt(filedirec, unpack=True)
        
    phase = (t/p)%1
    phase = T_0_fixer(t, m, p, phase)
            
    df_OGLE = pd.DataFrame({'t': t-np.min(t), 
                            'm': m,
                            'e': e,
                            'phase': phase })
        
    return df_OGLE

def read_fits (filename, tp = 'RRLYR'):
    bands = ['LC_I', 'LC_V', 'LC_H', 'LC_K']
    filedirec = 'lc_example/' + filename + '.fits'
    hdul = fits.open(filedirec)
    for b in bands:
        try:
            t = hdul[b].data['HJD']
            m = hdul[b].data['mag']
            e = hdul[b].data['mag_error']
        except:
            print('No %s band was found.'%(b.split('_')[1]))
            pass    
    p_range = info[tp]['p_range']
    
    phase, period = phase_refine(t, m, e, p_range) 
    phase = T_0_fixer(t, m, p, phase)
    
    
    df_OGLE = pd.DataFrame({'t': t-np.min(t), 
                            'm': m,
                            'e': e,
                            'phase': phase})
    
    return period, df_OGLE

def find_large_gaps (df):
    t_diff = np.diff(df['t'].values)
    t_diff = np.concatenate((t_diff, [0]), axis=0)
    return df[t_diff>100]

def phase_refine(t, m, e, p_range):
    
    period = np.linspace(p_range[0]-(p_range[0]/10), p_range[1]+(p_range[1]/10), 100)
    
    ls = LombScargle(t, m, e, nterms = 5)

    f_steps = 1./period
    power_window = ls.power(f_steps)
    
    p = (period[np.argmax(power_window)])
    
    phase = (t/p)%1

    return phase, p

def prep_input(df, tp, p):
    
    n_phases = info[tp]['n_phs']
    
    if not np.isinf(n_phases):

        x, y, e = create_multi_phase_repeating_folded_lc(n_phases, p, df)
        y = y[x.argsort()]
        e = e[x.argsort()]
        x = x[x.argsort()]
        X = np.linspace(0, 1, 100)

    else:
        x, y, e = df.t.values, df.m.values, df.e.values
        y = y[x.argsort()]
        e = e[x.argsort()]
        x = x[x.argsort()]
        X = np.linspace(0, 1713, 100)

    return x, y, e, X

def create_X_from_n_periods(n, x_n, y_n, period, t_max):
    total_periods = int(((t_max/(period)) - (t_max/(period))%1)+1)
    num_x_n =  int(total_periods/n -total_periods/n%1 +1)
    
    X = []
    Y = []
    for i in range(num_x_n):
        X += list(x_n * period * n + (i * n * period))
        Y += list(y_n)
    X = np.asarray(X)
    Y = np.asarray(Y)

    
    return X, Y
    
def binning(x, y, e, bins = 100):
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y,\
                                                         statistic='mean', bins=100)
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

def gap_reducer(df, p):  
    
    indxs = find_large_gaps(df).index.values
    df_gapped = df.copy(deep=True)

    for ind in indxs:
        if ind+1 >= len(df):
            continue
        
        gap = ((df.t[ind+1]-df.t[ind])/p)
        
        if gap<1:
            pass
        else:
            coeff_diff = (((df.t[ind+1] - df.t[ind])/p) - 
                         ((df.t[ind+1] -df.t[ind])/p)%1)
            df.loc[df.index>ind, 't'] = df.t[ind+1:] - coeff_diff*p


    df = df.reset_index().drop('index', axis =1)
    
    df_temp = df.copy()
    
    df_temp.t = df_temp.t + ((df.t[len(df)-1]/p)- (df.t[len(df)-1]/p)%1)*p
    
    df = pd.concat([df, df_temp]) #df.append(df_temp)
    df = df[df.t < 1713]

    df = df.sort_values(by=['t'])
    
    
    df = df.reset_index().drop('index', axis =1)
    

    return df

def fix_sampling (df, t_new, p):
    
    # Set up the Gaussian process.
    kernel =  kernels.CosineKernel(np.log(p)) + kernels.Matern32Kernel(1500000)
    
    gp = george.GP(kernel, solver= george.HODLRSolver)
        
    # Pre-compute the factorization of the matrix.
    gp.compute(df.t, df.e.values)

    t_prime = np.linspace(0, max(t_new),1000)
    
    mu, cov = gp.predict(df.m, t_prime)
    std = np.sqrt(np.diag(cov))
    
    intpl = interp1d(t_prime, mu)
    m_new = intpl(t_new)
    
    noise_fun = noise_function('cycle6_snr_curve.txt')
    mu_err = noise_fun(m_new)
    
    
    
    
    df_new = pd.DataFrame({'t': t_new, 'm': m_new , 'e': mu_err})
    
    return df_new

def create_multi_phase_repeating_folded_lc(n_phases, period, df):
    n_chunks = 4
    points_per_chunk = 500
    tmp_t = []
    tmp_m = []
    tmp_e = []
    for i in range(n_chunks):
        if not i == n_chunks-1:
            df_new = df[i*points_per_chunk: (i+1)*points_per_chunk]
            tmp_t += ((df_new.t.values[i*500:(i+1)*500]/(n_phases*period))%1).tolist()
            tmp_m += (df_new.m.values[i*500:(i+1)*500]).tolist()
            tmp_e += (df_new.e.values[i*500:(i+1)*500]).tolist()
        else:
            df_new = df[i*points_per_chunk:]
            tmp_t += ((df_new.t.values[i*500:]/(n_phases*period))%1).tolist()
            tmp_m += (df_new.m.values[i*500:]).tolist()
            tmp_e += (df_new.e.values[i*500:]).tolist()
        
    x = np.asarray(tmp_t)
    y = np.asarray(tmp_m)
    e = np.asarray(tmp_e)
    return x,  y, e


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

    # print('smoothness is ', smoothness)

    # print (p, -ll if np.isfinite(ll) else 1e25)
    return -ll if np.isfinite(ll) else 1e25

def opt_gp(p0, gp, x, y, s =1):
    results = op.minimize(nll, [p0[0], p0[1]],
                          args=(y,
                                x, gp, s))
    gp.kernel.parameter_vector = results.x

    return gp

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