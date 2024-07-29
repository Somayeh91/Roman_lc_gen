This repo is made to share code on how to change the sampling of an OGLE light curve to sampling of the light curves for the Roman Space telescope.




The example in notebook "fill_gaps_single_run" shows how Gaussian processes with a periodic kernel are fit to an OGLE long-period variable (LPV) star's light curve to find the best non-parametric fit. Next, the time sampling is changed according to the time sampling of light curves published in the micrtolensing data challenge (https://www.microlensing-source.org/data-challenge/). Noise is added to the data based on the noise function published at https://github.com/mtpenny/wfirst-ml-figures/blob/master/tables/cycle6_snr_curve.txt.



![In this figure, you can see on the left panel the OGLE light curve of a LPV star that is copied to fit the longer baseline of Roman. The orange line is the GP model fit through the OGLE data. On the right panel, the GP model is used to resample the light curve into Roman's time sampling and the noise is added on top of that.](https://github.com/Somayeh91/Roman_lc_gen/blob/main/OGLE_Roman_full_lc_with_GP_model.pdf)


