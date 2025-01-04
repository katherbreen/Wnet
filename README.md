
# This repository has been moved to https://github.com/dbarahon/Wnet

# Vertical Velocity Standard Deviation Calculation and Plotting

## Overview

This repository contains four different Python scripts for calculating and plotting the standard deviation in vertical wind velocity (sigmaW), and to demonstrate the Constraining Adversarial Trainining method following Barahona et al. "Deep Learning Parameterization of Vertical Wind Velocity Variability via Constrained Adversarial Training (doi:https://doi.org/10.1175/AIES-D-23-0025.1)". These scripts are designed to work with gridded, netcdf data from observations, high resolution simulated output, and reanalysis data. Observartional and reanalysis data should represent time series of sigmaW and atmospheric state  at different vertical positions, respectively, while the simulated data is taken from the NAS GEOS-5 Nature Run (G5NR), https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/data_access/. Other scripts provide various methods for computing and visualizing sigmaW.

## Files

### 1. `Wnet_prior.py`

This scrip uses data from G5NR, to train and neural network, "Wnet-prior" that reads predicts sigmaW from the meteorological state. Because the G5NR data set is too extensive to fit in memory, only a few half-hourly output files (3-5 files) are loaded at once for training a few epochs. Then a entire new set is loaded and so on. This behavior is controlled by the parameters dtbatch_size and epochs_per_dtbatch. For training always on the same files set epochs_per_dtbatch > number epochs. If only a single "time step" from G5NR is used for training set dtbatch_size = 1. The weights for the latest of Wnet-prior can be found in the 'data' directory.   

Using dask, the training datasets are lazily loaded. A "dask-generator" class feeds data for training, aligning each minibatch with the chunks of the dask array.  After training the script produces the weights of the neural network, Wnet_prior.h5, and plots the loss functions. If test mode is enabled, then the script tests Wnet_prior on a set of randomly selected files and saves the results in Wnet_prior.nc.


### 2. `Wnet_GAN.py`

This script refines the predictions of the Wnet_prior neural network using conditinonal generative adversarial training. Wnet_prior acts sa the generator and a second NN is build to act as the discriminator. A GAN class and custom training loop are build to set the adversarial training. The data used to train the networks consist of time series of sigmaW collected from ground stations around the world and reanalysis data (MERRA-2, https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/) collocacted in time and space with the observations. Sample data for two representative sites can be found in the data directory. After training the generator and discriminator weights are saved, and the losses plotted; best_generator.h5 constitutes the weights of the Wnet parameterization.  


### 3. `BoxPlots.py`

This script creates box plots comparing predictions from different models at ground sites against observations. Representative data to run thew script as well as the latest versions of the different NN models cand be found in the 'data' directory.  


### 4. `Pdf_bysite.py`

Similar to BoxPlots.py but compares and plots the probability distribution functions predicted by different models and the observations at each site, as well as computing the Kolmogorov-Smirnov statistic. 


## Dependencies

Each script has specific dependencies, which can be installed using `pip` or another package manager. Please refer to the individual script's documentation for details on their dependencies. 

## Example Data

You can find example data in the `data` directory for testing these scripts. Feel free to use this data to get started.


## Acknowledgments

If you find these scripts useful, please consider giving credit by citing this repository, and the supporting paper, in your project's acknowledgments.

---

# Wnet

GEOS software: https://github.com/GEOS-ESM/GEOSgcm

Data availability:
<ul>
<li>GEOS-5 Nature Run: https://gmao.gsfc.nasa.gov/global_mesoscale/7km-G5NR/data_access/
<li>MERRA-2: doi: 10.5067/VJAFPLI1CSIV
<li>Observations: https://www.arm.gov/data/
