import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from george import kernels
import george

def noise_function(noise_snr_file):
    
    noise_roman = np.loadtxt(str(noise_snr_file))
    
    noise_intpl = interp1d(noise_roman[:,0], noise_roman[:,1])
    
    return noise_intpl

def read_OGLE_lc (filename, p):
    filedirec = 'lc_example/' + filename + '.dat'
    t, m, e = np.loadtxt(filedirec, unpack=True)
    
    t_ = t- min(t)
    m_ = m- np.median(m)
    
    phase = (t/p)%1
    
    t_diff = np.diff(t)
    
    df_OGLE = pd.DataFrame({'t': t[:-1], 
    						'm': m[:-1],
    						't_': t_[:-1], 
    						'm_': m_[:-1], 
    						'e': e[:-1],
    						't_diff': np.diff(t), 
    						'phase': ((t/p)%1)[:-1] })
    
    return df_OGLE

def find_large_gaps (df):
    return df[df.t_diff>100]


def fill_gaps (df_gapped, ind, p):
    
#     for i in reversed(range(len(df_large_gaps))):#len(df_large_gaps)
        
#         ind = df_large_gaps.index[i]
        
    df = df_gapped.copy(deep=True)
    
    coeff_diff = ((df_gapped.t_[ind+1] - df_gapped.t_[ind])/p) - ((df_gapped.t_[ind+1] - df_gapped.t_[ind])/p)%1
    #((df.t[ind+1]/p)- (df.t[ind+1]/p)%1 - ((df.t[ind]/p)- (df.t[ind]/p)%1))/2

    df.loc[df.index>ind, 't'] = df.t[ind+1:] - coeff_diff*p
    df.loc[df.index>ind, 't_'] = df.t_[ind+1:] - coeff_diff*p
         
            
    df = df.sort_values(by=['t'])
                    
    return df


def phase_refine(t, m, e, p):
	# Refining the period:

    # print( p)
    
    period = np.linspace(p-(p/10), p+(p/10), 100)

    # print(1./period)
    
    ls = LombScargle(t, m, e, nterms = 5)

    f_steps = 1./period
    power_window = ls.power(f_steps)
    
    p = (period[np.argmax(power_window)])
    
    phase = (t/p)%1

    return phase, p


def gap_remover(filename, p):
    
    filedirec = 'lc_example/' + filename + '.dat'
    t, m, e = np.loadtxt(filedirec, unpack=True)
    
    t = t- min(t)
#     m = m- np.median(m)
    
    
    phase = (t/p)%1
 #phase_refine(t, m, e, p)
    
    t_diff = np.diff(t)
    
    df_gapped = pd.DataFrame({'t': t[:-1], 'm': m[:-1], 'e': e[:-1], 'phase': phase[:-1] })
    
    indxs = df_gapped[np.diff(t)>50].index.values


    df = df_gapped.copy(deep=True)

    for ind in indxs:
        if ind+1 >= len(df_gapped):
            continue
        
        gap = ((df_gapped.t[ind+1]-df_gapped.t[ind])/p)
        
        if gap<1:
            # print(gap) 
            # print('The gap is not larger than one period.')
            # df = df_gapped
            pass
        else:

            

            

            coeff_diff = ((df_gapped.t[ind+1] - df_gapped.t[ind])/p) - ((df_gapped.t[ind+1] - df_gapped.t[ind])/p)%1

            df.loc[df.index>ind, 't'] = df.t[ind+1:] - coeff_diff*p


    df = df.reset_index().drop('index', axis =1)
    
    df_temp = df.copy()
    
    df_temp.t = df_temp.t + ((df.t[len(df)-1]/p)- (df.t[len(df)-1]/p)%1)*p
    
    df = df.append(df_temp)
    df = df[df.t < 1713]
    
            
    
    df = df.sort_values(by=['t'])
    
    
    df = df.reset_index().drop('index', axis =1)
    

    return df_gapped, df, p

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

def create_folded_lc(n_phases, df):
	x = pd.concat([df.sort_values(by=['phase']).phase, 
               df.sort_values(by=['phase']).phase+1,
              df.sort_values(by=['phase']).phase+2])
	y = pd.concat([df.sort_values(by=['phase']).m, 
	               df.sort_values(by=['phase']).m,
	              df.sort_values(by=['phase']).m])
	e = pd.concat([df.sort_values(by=['phase']).e, 
	               df.sort_values(by=['phase']).e,
	              df.sort_values(by=['phase']).e])
	return x,  y, e

