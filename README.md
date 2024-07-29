This repo is made to share code on how to change the sampling of an OGLE light curve to sampling of the light curves for the Roman Space telescope.




The example in notebook "fill_gaps_single_run" shows how Gaussian processes with a periodic kernel are fit to an OGLE long-period variable star's light curve to find the best non-parametric fit. Next, the sampling is changed according to the sampling of light curves published in the micrtolensing data challenge (https://www.microlensing-source.org/data-challenge/). Noise is added to the data based on the noise function published at https://github.com/mtpenny/wfirst-ml-figures/blob/master/tables/cycle6_snr_curve.txt.
