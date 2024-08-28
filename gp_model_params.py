import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from george import kernels
import george
import scipy.optimize as op

info = {'LPV':{'kernel':kernels.CosineKernel(np.log(1))+
                        kernels.Matern32Kernel(15),
               'p0':[1, 15], 
               'n_phs': np.inf,
               'p_range': [1, 500],
               'fit_binned': False,
               'gp_opt': False,
               'p0_period': lambda x: np.log10(x),
               'example': 'OGLE-BLG-LPV-048338'},
        'RRLYR':{'kernel':kernels.CosineKernel(np.log(1))+kernels.Matern32Kernel(1.5),
                 'p0':[1, 1.5], 
                 'n_phs': 10,
                 'p_range': [0.001, 10],
                 'fit_binned': True,
                 'gp_opt': True,
                 'p0_period': lambda x: np.log10(1/x),
                 'example': 'OGLE-BLG-RRLYR-03350'},
        'DSCT':{'kernel','p0', 'n_phs'},
        'ECL':{'kernel','p0', 'n_phs'},
        'ELL':{'kernel','p0', 'n_phs'},
        'T2CEP':{'kernel':kernels.CosineKernel(np.log(1))+kernels.Matern32Kernel(1.5),
                 'p0':[1, 1.5], 
                 'n_phs': 10,
                 'p_range': [0.001, 10],
                 'fit_binned': False,
                 'gp_opt': True,
                 'p0_period': lambda x: np.log10(1/x),
                 'example': 'OGLE-BLG-TC2CEP-078'},
        'HB':{'kernel','p0', 'n_phs'}
        }