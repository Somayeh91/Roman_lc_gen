def noise_function(noise_snr_file):
    
    noise_roman = np.loadtxt(str(noise_snr_file))
    # if min(noise_roman[:,0])<min_ and max(noise_roman[:,0])>max_:
    noise_intpl = interp1d(noise_roman[:,0], noise_roman[:,1])
#     elif min(noise_roman[:,0])<min_ and max(noise_roman[:,0])<max_:
#         ind = noise_roman[:,0]<
#         noise_intpl = interp1d(noise_roman[,0], noise_roman[:,1])
    
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

def find_large_gaps (df):
    t_diff = np.diff(df['t'].values)
    t_diff = np.concatenate((t_diff, [0]), axis=0)
    return df[t_diff>100]

def fix_sampling(df, t_new, p):
    
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

def checkbounds(x_old, x_new):
    if np.min(x_new)<np.min(x_old):
        return False
    elif np.max(x_new)>np.max(x_old):
        return False
    else:
        return True

def x(t_OGLE, mu_OGLE, t_Roman, mu_Roman):
    intpl = interp1d(t_Roman, mu_Roman)
    m_new = intpl(t_OGLE)
    metrics_tmp = np.nansum((m_new-(mu_OGLE))**2)/len(mu_OGLE)
    return metrics_tmp

def get_metric(y_binned, mu_tmp):
    return np.nansum((y_binned-(mu_tmp))**2)/len(y_binned)


def check_fit(x_binned, gp_y_binned_fit, regular_sampling_fit, gp_y_regular_fit):

    gp_y_regular_fit = gp_y_regular_fit[(regular_sampling_fit>min(x_binned)) & (regular_sampling_fit<max(x_binned)) ]
    regular_sampling_fit = regular_sampling_fit[(regular_sampling_fit>min(x_binned)) & (regular_sampling_fit<max(x_binned)) ]
    intpl = interp1d(x_binned, gp_y_binned_fit)
    m_new = intpl(regular_sampling_fit)

    return np.sum((m_new-gp_y_regular_fit)**2)/len(regular_sampling_fit)


#old data binning process:
# Binning according to number of data points
# note that min and max of regular_sampling_fit and x_binned are the same
# if len(x)>1000 and info[tp]['fit_binned']==False:
#     info[tp]['fit_binned'] = True
#     info[tp]['n_bins'] = 500
    
# if len(x)/10>10000:
#     x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, y, e, info[tp]

#         , n_bins=int(len(x)/1000))
# elif len(x)/10>1000 and len(x)/10<10000:        
#     x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, y, e, info[tp], n_bins=int(len(x)/100))
# elif len(x)/10>100 and len(x)/10<1000: 
#     x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, y, e, info[tp], n_bins=int(len(x)/10))
# elif len(x)/10<100 and len(x)/10>10:
#     x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, y, e, info[tp], n_bins=int(len(x)/5))
# elif len(x)<100:
#     info[tp]['fit_binned'] = False
#     x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, y, e, info[tp], n_bins=np.nan)
# else:
#     x_binned, y_binned, e_binned, regular_sampling_fit = binning_data(x, y, e, info[tp], n_bins=np.nan)