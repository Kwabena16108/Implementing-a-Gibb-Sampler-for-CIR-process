#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 17:27:31 2021

@author: dicksonnkwantabisa
"""


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import seaborn as sns
from math import inf
import time as time
from IPython import get_ipython
from datetime import datetime as datetime
get_ipython().run_line_magic('matplotlib', 'inline')


#%% Simulation of Interest rate 
def inst_to_ann(r):
    """
    Converts short rate to annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Converts annualized to short rate
    """
    return np.log1p(r)


# the parameters are from the Bayesian CIR estimation

alpha = params['alpha'].to_numpy().flatten(); 
beta = params['beta'].to_numpy().flatten(); sig2 = params['sigma2'].to_numpy().flatten()
theta = alpha.mean() / beta.mean()
kappa= alpha.mean()
sig2 = sig2.mean()

a = alpha.mean() ; b = alpha.mean() / beta.mean() ; sigma = np.sqrt(sig2.mean())


# Simulation of Bond prices zero-coupon

def cir_price(n_years, n_scenarios, a, b, sigma, steps_per_year, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well.
    """
    if r_0 is None: r_0 = b 
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # int because n_years might be a float
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    ####
    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])
        
    rates = pd.DataFrame(data=(rates), index=range(num_steps))
    ### for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
  
    return rates, prices


zc_bond = cir_price(n_years=1., n_scenarios=100, a=a, b=b, sigma=sigma, steps_per_year=100)
zc_bond[1].plot(legend=False, figsize=(12,5), ylabel='bond price');

plt.figure(figsize=(10,6))
plt.plot(zc_bond[0], lw=1.)
plt.xlabel('time(10 years horizon)')
plt.grid(True)
plt.axhline(y=b, color='black', linestyle='--')
plt.ylabel('interest rate level');

sns.distplot(zc_bond[0][-1], hist=False)

