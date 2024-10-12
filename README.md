# Graph-based Methods for Forecasting Realized Covariances

This is the README file for the project "Graph-based Methods for Forecasting Realized Covariances". It provides an overview of the project structure and instructions on how to use and contribute to the codebase.

## Table of Contents

- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data](#data)
- [Computing Environment](#computing-environment)

## Project Structure

The project is organized as follows:
- `GHAR_Var.py`: Linear models to forecast the realized volatility, including HAR and GHAR. HAR is a special case of GHAR, assuming the adjacency matrix is identity.
- `GHAR_Corr.py`: Linear models to forecast the realized correlation
- `GHAR_DRD.py`: Combine the forecasted variance and correlation to get the forecasted covariance matrix
- `GMVP.py`: Use forecasted covariance matrix to compute the GMVP portfolio, and record its performance
- `GMVP+.py`: Use forecasted covariance matrix to compute the GMVP+ portfolio, and record its performance
- `MCS.py`: Implementation of Econometrica Paper: "The model confidence set." by Hansen, Peter R., Asger Lunde, and James M. Nason. 
- `Stats_MCS.py`: Summarize the results of the forecast models, including the Euclidean, Frobenius, QLIKE, and the MCS tests.


## Data
The data used in this reproducibility check is LOBSTER (https://lobsterdata.com/), which needs to be purchased by users.

## Computing Environment
To run the reproducibility check, the following computing environment and package(s) are required:
- Environment: These experiments were conducted on a system equipped with an Nvidia A100 GPU with 40 GB of GPU memory, an AMD EPYC 7713 64-Core Processor @ 1.80GHz with 128 cores, and 1.0TB of RAM, running Ubuntu 20.04.4 LTS. 

- Package(s): 
    - Python 3.8.18
    - numpy 1.22.3
    - pandas 2.0.3
    - scikit-learn 1.3.0
    - matplotlib 3.7.2
