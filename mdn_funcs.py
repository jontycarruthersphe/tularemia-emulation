# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:42:55 2023

@author: Jonathan.Carruthers
"""

# Functions used for testing the performance of the trained MDN

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from itertools import combinations
import pandas as pd
from scipy.optimize import newton


def softmax(W, t=1.0):
    """
    W: array of logits of size (# test sets x # mixes)
    t: the temperature to adjust the distribution (default 1.0)
    """
    n_test = W.shape[0]
    E = W / t   # adjust temperature
    E = W - np.max(W, axis=1).reshape((n_test,1))   # subtract max to protect from exploding exp values
    E = np.exp(E)
    dist = E / E.sum(axis=1).reshape((n_test,1))
    return dist


def mixture_pdf(xx, prob, mean, sd):
    """
    xx: array of values to evaluate pdf at
    prob: array of weights for mixture distribution (# mixes,)
    mean: array of means for mixture components (# mixes,)
    sd: array of standard deviations for mixture components (# mixes,)
    """
    n_mixes = prob.shape[0]
    individual_pdfs = [st.norm.pdf(xx,loc=mean[i],scale=sd[i]) for i in range(n_mixes)]
    return np.dot(prob, individual_pdfs)


def mixture_cdf(xx, prob, mean, sd):
    """
    xx: array of values to evaluate pdf at
    prob: array of weights for mixture distribution (# mixes,)
    mean: array of means for mixture components (# mixes,)
    sd: array of standard deviations for mixture components (# mixes,)
    """
    n_mixes = prob.shape[0]
    individual_cdfs = [st.norm.cdf(xx,loc=mean[i],scale=sd[i]) for i in range(n_mixes)]
    return np.dot(prob, individual_cdfs)


def mixture_sample(prob, mean, sd, n_sample):
    """
    Get a random sample of size n_sample from the mixture distribution
    """
    mix_ind = np.random.choice(len(prob), p=prob, replace=True, size=n_sample)
    return st.norm.rvs(loc=mean[mix_ind], scale=sd[mix_ind])


def KS_score(y_test, pis, mus, sigs):
    """
    y_test: array of test set response times (# test sets x sample size)
    pis: array of weights for all mixture distributions (# test sets, # mixes)
    mus: array of means for all mixture distributions (# test sets, # mixes)
    sigs: array of standard deviations for all mixture distributions (# test sets, # mixes)                                                         
    """
    n_sets = y_test.shape[0]
    KS = np.zeros((n_sets,1))
    for i in range(n_sets):
        ks_stat, p_value = st.kstest(y_test[i], mixture_cdf, args=(pis[i], mus[i], sigs[i]))    
        KS[i] = ks_stat
    return KS


def predicted_i95(prob, mean, sd):
    """
    Estimate the inter-95-percentile range of the mixture distribution using a
    Newton root-finder with initial guesses based on the inter-95-percentile
    range of a random sample
    """
    sample = mixture_sample(prob, mean, sd, 10000)
    guess_low = np.percentile(sample, q=2.5)
    guess_high = np.percentile(sample, q=97.5)
    try:
        low = newton(lambda x: mixture_cdf(x, prob, mean, sd)-0.025, x0=guess_low, maxiter=500)
        low_approx = False
    except:
        low = guess_low
        low_approx = True
    try:
        high = newton(lambda x: mixture_cdf(x, prob, mean, sd)-0.975, x0=guess_high, maxiter=500)
        high_approx = False
    except:
        high = guess_high
        high_approx = True
    return high - low, (low_approx or high_approx)


def scaled_KS_score(y_test, pis, mus, sigs):
    """
    Get the scaled KS test statistics that account for the width of the distribution
    """
    # problem specific parameters for the scaling
    epsilon = 0.2
    k = 0.8
    phi = 5
    
    n_sets = y_test.shape[0]
    KS_scaled = np.zeros((n_sets,1))
    R_approx = np.zeros(n_sets)
    for i in range(n_sets):
        # original KS test statistic
        ks_stat, p_value = st.kstest(y_test[i], mixture_cdf, args=(pis[i], mus[i], sigs[i]))
        
        # inter-95-percentile range
        predicted_R, approx = predicted_i95(pis[i], mus[i], sigs[i])
        R_approx[i] = approx
        test_R = np.percentile(y_test[i], q=97.5) - np.percentile(y_test[i], q=2.5)
        R = max(predicted_R, test_R)
        
        # scaled KS test statistic
        scale_factor = 1 - epsilon / ((1+np.exp(phi*(ks_stat-k))) * (epsilon+R))
        KS_scaled[i] = ks_stat * scale_factor
    return KS_scaled, R_approx.sum()
        
        
def set_step(r):
    """
    Set the bin width for a histogram based on the range of values being plotted
    r: range of values being plotted
    """
    if r < 0.1:
        return 0.002
    elif r < 0.5:
        return 0.01
    elif r < 2:
        return 0.1
    elif r < 5:
        return 0.25
    else:
        return 0.5
    

def mixture_bounds(prob, mean, sd, low, high):
    """
    Get suitable lower and upper bounds for plotting the mixture distribution
    low: initial guess at lower bound
    high: initial guess at upper bound
    """
    while mixture_cdf(low, prob, mean, sd) > 0.01:
        low -= 0.01
    while mixture_cdf(high, prob, mean, sd) < 0.99:
        high += 0.01
    return low, high
    

def plot_predictions(y_test, prob, mean, sd):
    """
    Plot comparing the pdf of the mixture distribution with a histogram of the 
    response times from the test set.
    y_test: array of test set response times (# test sets x sample size)
    prob: array of weights for mixture distribution (# mixes,)
    mean: array of means for mixture components (# mixes,)
    sd: array of standard deviations for mixture components (# mixes,)
    """
    # getting the KS-statistic between the predicted cdf and true sample
    ks_stat, p_value = st.kstest(y_test, mixture_cdf, args=(prob, mean, sd))
    
    # sorting out the ranges and bins for plotting
    x_min, x_max = y_test.min(), y_test.max()
    mix_low, mix_high = mixture_bounds(prob, mean, sd, x_min, x_max)
    x_min, x_max = min(x_min, mix_low), max(x_max, mix_high)
    step = set_step(x_max - x_min)
    bin_min = np.floor(x_min/step)*step
    bin_max = np.ceil(x_max/step)*step
    bins = np.arange(bin_min, bin_max+step, step)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    # histogram of true values from the test set
    axes[0].hist(y_test, density=True, bins=bins, color="C0", alpha=0.5, label="True")
    
    # individual mixture component with weight > 0.1
    n_mixes = prob.shape[0]
    for i in range(n_mixes):
        #if prob[i] > 0.1:
        if prob[i] > 0.01:
            Z = st.norm(loc=mean[i], scale=sd[i])
            zz = np.linspace(bin_min, bin_max, 1000)
            axes[1].plot(zz,Z.pdf(zz),color="C%s"%i, alpha=0.5, 
                         label="$\\pi$=%.2f, $\\mu$=%.2f, $\\sigma$=%.2f"%(prob[i], mean[i], sd[i]))
    
    # plotting the mixture distribution
    mix = mixture_pdf(zz, prob, mean, sd)
    axes[0].plot(zz, mix, color="C2", label="Predicted")
    axes[1].plot(zz, mix, color="k", label="Mixture")
    axes[0].set_title("KS test statistic: %.2f"%ks_stat)
    axes[0].set_xlabel("Time to response",fontsize=14)
    axes[0].set_ylabel("Density",fontsize=14)
    axes[1].set_xlabel("Time to response",fontsize=14)
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()


def KS_input_space(x_test, ks):
    """
    Scatter plot indicating areas of input space with high KS test statistics
    x_test: array of test inputs (# test sets x # inputs)
    ks: array of KS test statistics (# test sets, )                             
    """
    KS_df = pd.DataFrame(np.hstack((x_test, ks)), columns=["alpha","mu/alpha","dose","M","KS"])
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))
    axes = axes.flatten()
    cmap = sns.color_palette("Blues", as_cmap=True)
    norm = plt.Normalize(0, 1)
    for i, pair in enumerate(combinations(["alpha", "mu/alpha", "dose", "M"], 2)):
        sns.scatterplot(x=pair[0],y=pair[1],data=KS_df, hue="KS", palette=cmap, size=1,
                        ax=axes[i + i//3])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        axes[i + i//3].get_legend().remove()
    
    fig.delaxes(axes[3])
    fig.delaxes(axes[7])
    cbar_ax = fig.add_axes([0.8, 0.15, 0.025, 0.75])
    fig.colorbar(sm, cax=cbar_ax)
    fig.axes[-1].set_ylabel("KS test statistic", fontsize=15, labelpad=10)
    plt.tight_layout()


def KS_histogram(ks):
    """
    ks: array of KS test statistics
    """
    plt.figure()
    n_sets = ks.shape[0]
    plt.hist(ks, bins=np.arange(0,1.05,0.05))
    plt.title("%d test sets"%(n_sets), fontsize=14)
    plt.xlabel("KS test statistic", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()


def KS_bivariate_hist(x_test, ks, par_ind):
    """
    x_test: array of test inputs (# test sets x # inputs)
    ks: array of KS test statistics (# test sets, )
    par_ind: column index of parameter in x_test to plot against KS test statistic
    """
    KS = np.hstack((x_test, ks))
    par_bins = np.arange(0, 1.1, 0.1)
    ks_bins = np.arange(0, 1.05, 0.05)
    
    KS_bi_hist = np.zeros((len(ks_bins)-1, len(par_bins)-1))
    for j in range(len(par_bins)-1):
        KS_split = KS[(KS[:,par_ind] >= par_bins[j]) & (KS[:,par_ind] < par_bins[j+1])]
        n_KS_split = KS_split.shape[0]
        KS_frac = [((KS_split[:,-1] >= ks_bins[i]) & (KS_split[:,-1] < ks_bins[i+1])).sum()/n_KS_split for i in range(len(ks_bins)-1)]
        assert np.isclose(sum(KS_frac), 1, rtol=0, atol=1e-10)
        KS_bi_hist[:,j] = KS_frac
    
    par_names = ["alpha","mu/alpha","dose","M"]
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    im = ax.imshow(KS_bi_hist.T, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(-.5, len(ks_bins)-1, 2))
    ax.set_xticklabels([str(i/10) for i in range(11)])
    ax.set_yticks(np.arange(-.5, len(par_bins)-1, 1))
    ax.set_yticklabels([str(i/10) for i in range(11)])
    ax.set_xlabel("KS test statistic", fontsize=14)
    ax.set_ylabel(par_names[par_ind],fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
        
    