# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:36:46 2023

@author: Jonathan.Carruthers
"""

import mdn
import emcee
import numpy as np
import pandas as pd
from tensorflow import keras
from scipy.stats import norm
from scipy.integrate import trapezoid
from pathlib import Path

from multiprocessing import Pool
from multiprocessing import set_start_method

import os
os.environ["OMP_NUM_THREADS"] = "1"

import mdn_funcs


def normalise(X, lower, upper):
    ''' normalise the inputs, assuming they are already on a log scale '''
    assert (X >= lower).all() & (X <= upper).all()
    X_norm = (X - lower) / (upper - lower)
    return X_norm


def block_cdf(x, mean, sd, weights):
    '''
    x (float): point to evaluate mixture cdf at
    mean, sd, weights ((n_M x N_MIXES) array): parameters of mixture distribution
    '''
    Z = norm(loc=mean, scale=sd)
    cum = Z.cdf(x)
    return (weights*cum).sum(axis=1)


def integrate_over_M(mean, sd, weights, M_pdf, obs, h):
    '''
    Evaluate the probability of observing an incubation period within a specific
    window, accounting for the distribution of the threshold, M.

    mean, sd, weights: (n_M x N_MIXES) arrays containing output for all values 
                        of M for a specific individual
    M_pdf: (n_M,) array containing the density of M
    obs: (2,) array of the start/end points of the incubation period window
    h: float, the step size between values of M
    '''
    F_early = block_cdf(obs[0], mean, sd, weights)
    F_late = block_cdf(obs[1], mean, sd, weights) 
    F_diff = F_late - F_early
    integrand = F_diff * M_pdf
    integral = trapezoid(integrand, dx=h)
    return integral


def emulator_prob_prediction(theta):
    '''
    Evaluate the probability of illness using the emulator
    
    theta: (3,) array containing the current estimate of alpha, mu/alpha and E[M]
    '''
    # stack the (alpha, mu/alpha) values along with the correct doses and normalise
    pars = np.repeat(theta[:2],n_responses).reshape((n_responses,2),order='F')
    X = np.hstack((pars, log10_prob_doses))
    X_norm = normalise(X, prob_input_lower, prob_input_upper)
    prob = prob_loaded_model.predict(X_norm, verbose=0).flatten()
    return prob


def emulator_time_prediction(theta):
    ''' 
    Evaluate the probability of the observed incubation period for each individual
    
    theta: (3,) array containing the current estimate of alpha, mu/alpha and E[M]
    '''
    # Density of M given its mean
    h = 0.1
    M_dist = norm(loc=theta[2], scale=1)
    M_low = np.floor(M_dist.ppf(0.001)/h) / (1/h)
    M_high = np.ceil(M_dist.ppf(0.999)/h) / (1/h)
    M = np.linspace(M_low, M_high, 1+int((M_high-M_low)/h))
    M_density = M_dist.pdf(M)
    n_M = len(M)

    # stack the inputs into an (n_times*n_M, 4) array and normalise
    pars = np.repeat(theta[:2],n_times*n_M).reshape((n_times*n_M,2),order='F')
    M_rep = np.tile(M, n_times).reshape((n_times*n_M,1))
    dose_rep = np.repeat(log10_time_doses, n_M).reshape((n_times*n_M,1))
    X = np.hstack((pars, dose_rep, M_rep))
    X_norm = normalise(X, time_input_lower, time_input_upper)
    
    # evaluate the emulator at the inputs
    Y = time_loaded_model.predict(X_norm, verbose=0)
    mix_means = Y[:,:N_MIXES]
    mix_sds = Y[:,N_MIXES:2*N_MIXES]
    mix_weights = mdn_funcs.softmax(Y[:,2*N_MIXES:])
    
    prob_obs = [integrate_over_M(mix_means[n_M*i:n_M*(i+1),:], 
                                 mix_sds[n_M*i:n_M*(i+1),:],
                                 mix_weights[n_M*i:n_M*(i+1),:],
                                 M_density, log_incubation[i], h) for i in range(n_times)]
    return np.array(prob_obs)


def log_prior(theta):
    if (theta >= prior_lower).all() & (theta <= prior_upper).all():
        #log(1)=0 and we can igore proportional constants
        return 0
    return -np.inf


def log_likelihood(theta):
    F_diff = emulator_time_prediction(theta)
    p = emulator_prob_prediction(theta)
    logL_time = np.log(F_diff).sum()
    logL_prob = np.dot(responses, np.log(p)) + np.dot(1-responses, np.log(1-p))
    return logL_time + logL_prob


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    logL = log_likelihood(theta)
    if not np.isfinite(logL):
        return -np.inf
    return lp + logL


def find_start(n_guesses=500):
    sample = np.zeros((n_guesses, 4))
    for i in range(n_guesses):
        pars = np.random.uniform(prior_lower, prior_upper)
        logL = log_likelihood(pars)
        sample[i] = np.hstack((pars, logL))
    print(sample)
    return sample[np.nanargmax(sample[:,-1])][:-1]
    
    

#%%

# Read the Saslaw and Sawyer response time data and combine
sawyer = pd.read_excel('whitecoat_data/sawyer_incubation_period.xlsx')
saslaw = pd.read_excel('whitecoat_data/saslaw_incubation_period.xlsx')

combined = pd.concat((saslaw, sawyer)).reset_index(drop=True)
time_doses = combined['dose'].to_numpy()
n_times = len(time_doses)
log10_time_doses = np.log10(time_doses).reshape((n_times,1))

# the emulator outputs incubation periods in log-hours. We also assume that for
# a reported incubation period, the true time lies within the proceeding 24 hours
incubation_days = combined['incubation_period'].to_numpy().reshape((n_times,1))
incubation_interval = np.hstack(((incubation_days-1)*24, incubation_days*24))
log_incubation = np.log(incubation_interval)

# Read in the probability of response data
whitecoat = pd.read_excel('whitecoat_data/prob_illness.xlsx')
prob_doses = whitecoat['dose'].to_numpy()
n_responses = len(prob_doses)
log10_prob_doses = np.log10(prob_doses).reshape((n_responses,1))
responses = whitecoat['response'].to_numpy()

# Load the emulator for the incubation period
N_MIXES = 10
time_loaded_model = keras.models.load_model('final_emulators/emulator_time.h5',
                                            custom_objects={"MDN": mdn.MDN, 
                                                            "mdn_loss_func": mdn.get_mixture_loss_func(1, N_MIXES)})
# Load the emulator for the probability of illness
prob_loaded_model =  keras.models.load_model('final_emulators/emulator_prob.h5')

# bounds for normalising inputs of the response time emulator
time_input_lower = np.array([-3, np.log10(1/40), 0, 5])
time_input_upper = np.array([0, np.log10(40), 5, 20])

# bounds for normalising inputs of the prob response emulator
prob_input_lower = np.array([-3, np.log10(1/40), 0])
prob_input_upper = np.array([0, np.log10(40), np.log10(250)])

# bounds for the prior distributions (alpha, mu/alpha and E[M])
prior_lower = np.array([-3, np.log10(1/40), 8.25])
prior_upper = np.array([0, np.log10(40), 16.75])


# running the mcmc sampling
nwalkers = 32
ndim = 3
n_steps = 500
continue_sampling = False

# for saving the sampler
path = Path('samplers')
path.mkdir(exist_ok=True)
backend = emcee.backends.HDFBackend(path / 'whitecoat_inference.h5')
if not continue_sampling:
    backend.reset(nwalkers, ndim)
    
if __name__=='__main__':
    set_start_method('spawn')
    print(f'Starting with size {backend.iteration}')
    with Pool(processes=4) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        pool=pool, backend=backend)
        if continue_sampling:
            sampler.run_mcmc(None, n_steps, progress=True)
        else:
            # start the walkers off around probable parameters from a grid search
            start = find_start()
            theta0 = start + 1e-3 * np.random.randn(nwalkers, ndim)
            sampler.run_mcmc(theta0, n_steps, progress=True)
    print(f'\nEnding with size {backend.iteration}')


