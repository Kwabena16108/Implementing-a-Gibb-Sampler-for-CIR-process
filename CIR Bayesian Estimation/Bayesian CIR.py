#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:45:43 2021

@author: dicksonnkwantabisa
"""


import numpy as np
import pandas as pd
import math
from scipy.stats import truncnorm
from scipy.stats import invgamma
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random
import seaborn as sns
import math
from math import inf
import time as time
from IPython import get_ipython
from datetime import datetime as datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# read in data
filename = 'Canada Interest Rate.csv'
data = pd.read_csv(filename,usecols=['rates'])
data = data.to_numpy()
data = data.flatten()

# data augmentation
T_=len(data)           # time (in years) over which data was taken
print("Data points taken every month over 20 years\n")

M_=20 # length of augmented data
print("Augmenting ",M_," points between every pair of data point")


# priors used for data augmentation assumed to be generated from a truncated normal distribution
s=0.2
sig2=truncnorm(a=-1/s,b=math.inf,loc=sig2,scale=s*sig2).rvs(); sig2
alpha0=truncnorm(a=-1/s,b=math.inf,loc=alpha,scale=s*alpha).rvs(); alpha0
beta0=truncnorm(a=-1/s,b=math.inf,loc=beta,scale=s*beta).rvs() ; beta0 

# alpha0=0.5; beta0=0.2; sig2_0=0.1;
print("Augmenting data with parameters: alpha = ",alpha0," beta = ",beta0," sigma2 = ",sig2_0)

random.seed(123)

# Now generate the augmented data
delta=1/252   # generate data at intervals of delta
R_=np.zeros((T_-1,M_))
R_[:,0]=data[:-1]         # initialize first column to given data, columns>1 will be augmented data
for j in range(M_-1):
    R_[:,j+1]=R_[:,j]+(alpha0-beta0*R_[:,j])*delta+np.sqrt(sig2_0*delta*R_[:,j])*np.random.standard_normal(size=T_-1)    # CIR process
    rst=np.append(R_.flatten(),data[-1])               # this is the augmented data


rst[rst<0] #checking for negative numbers

 # save augmented data
aug = pd.DataFrame(rst)
aug.to_csv('samples_data.csv')


 # define functions for each parameter of the CIR model

def update_psi2(rst, sig2, prior_cov, prior_mu):
    
    #updating alpha and beta....
    n = len(rst)
    y_bar = np.mean(rst)
    
    A=np.sum(1/rst[:-1])
    B=np.sum(rst[:-1])
  
    a11=delta*A/sig2
    a22=delta*B/sig2
    a12=-(delta*(T_-1)*(M_+1))/sig2
    
    invLambda = np.linalg.inv(np.array([[a11,a12],[a12,a22]]))
    invLambda_0 = np.linalg.inv(prior_cov)
    
   
    # generate new alpha and beta from updates from a trunc. multivariate normal
    
    amin, amax = 0, math.inf
    bmin, bmax = 0, math.inf
    
    mean = ((invLambda_0.dot(prior_mu) + n * y_bar * invLambda) * np.linalg.inv(invLambda_0 + n * invLambda)).diagonal()
    covariance = np.linalg.inv(invLambda_0 + n * invLambda)
    
    z = np.random.multivariate_normal(mean = mean, cov = covariance, size=(20))
    accepted = z[(np.min(z - [amin, bmin], axis=1) >= 0) & (np.max(z - [amax, bmax], axis=1) <= 0)]
    
    alpha, beta = np.mean(accepted, axis=0)
    
    return alpha, beta



def update_sig2(rst, nu_0, beta_0, alpha, beta):
    
    #updating sigma.....
    E = (T_-1)*(M_+1)/2
    nu_1 = nu_0 + E
    temp = ((rst[1:]-(rst[:-1]+(alpha-beta*rst[:-1])*delta))**2)/(2*rst[:-1])
    F = temp.sum()
    beta_1 = beta_0 + F
    
    # generate new sig2 from updates
    #sig2 = invgamma(nu_1, beta_1).rvs()
    sig2 = 1 / np.random.gamma(shape = nu_1, scale = 1 / beta_1)
    
    return sig2




 # define the Gibbs sampler

def gibbs_sampler(rst, hypers,n_iter, init):
    
    M = 2
    #matrix to hold params
    params = np.zeros((n_iter, 3))
    s_ = np.zeros((len(rst)-1, M))
    s = np.zeros_like(s_)
    s_[:, 0] = rst[:-1]
    
    #intialize algo
    sig2_now = init[0]
    #ir_now = init[2]
    alpha_now , beta_now  = init[1]
    
    ## begin Gibbs
    for i in range(n_iter):
        
        # updating params.....
        alpha_now , beta_now = update_psi2(rst = rst, sig2 = sig2_now, prior_cov = hypers[2], prior_mu = hypers[3])
        sig2_now = update_sig2(rst = rst, nu_0=hypers[0], beta_0=hypers[1], alpha=alpha_now, beta=beta_now)
        params[i, : ] = np.array((alpha_now, beta_now, sig2_now))
        
        #updating data itself with new params....
        for j in range(M-1):
            s_[:, j+1] = s_[:, j] + (alpha_now - beta_now * s_[:, j]) * delta + np.sqrt(sig2_now * delta * s_[:, j]) * np.random.standard_normal()
            s = s_

    # save updated params
    params = pd.DataFrame(params)
    params.columns = ['alpha', 'beta', 'sigma2']
    samples = pd.DataFrame(s)
    
    return samples, params


# Create priors

prior_cov = np.array([[10, 0.08], [0.08, 10]])
prior_mu = np.array([0.01621632, 0.02818355])
hypers = [0.01, 0.2, prior_cov, prior_mu]


# Create initializing values
delta=1/252
init = [0.001, [2, 0.45], 0.105]
n_iter = 15000

gibbs = gibbs_sampler(rst=rst, n_iter=n_iter, hypers=hypers, init=init)

gibbs[0].to_csv('post_mcmc.csv')
gibbs[1].to_csv('post_params.csv')

post_ir = gibbs[0]; params = gibbs[1]

params.agg(['min', 'max', 'mean', 'std']).round(3)


# plot results
trace_burnt = params
hist_plot = trace_burnt.hist(bins = 30, layout = (1,3), figsize=(12,6))

plt.figure(figsize=(12,12))
plt.subplot(311)
plt.plot(params['alpha'], lw=.05, label='alpha')
plt.plot(params['alpha'], 'b')
plt.ylabel('value')
plt.title('Trace of alpha')
plt.subplot(312)
plt.plot(params['beta'], lw=.05, label='beta')
plt.plot(params['beta'], 'b')
plt.ylabel('value')
plt.title('Trace of beta')
plt.subplot(313)
plt.plot(params['sigma2'], lw=.05, label='sigma2')
plt.plot(params['sigma2'], 'b')
plt.ylabel('value')
plt.xlabel('iterations')
plt.title('Trace of sigma2');

 # save results
gibbs[0].to_csv('post_mcmc.csv')
params.to_csv('post_params.csv')

