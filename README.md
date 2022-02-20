# Implementation of a Gibb-Sampler for CIR process

The following code implements the CIR (Cox, Ingersoll and Ross, 1985) stochastic interest rate model via a Gibb Sampler 
algorithm and simulate bond prices in Python.

The process involves finding the full conditional posterior distribution of model parameters in closed form.
The Gibb Sampler (an MCMC algorithm) is used to draw samples from the full conditional posterior distribution.
Readers can refer to https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3998313 for my working paper titled 
"A Bayesian Approach to Pricing Catastrophe Bond Derivatives " for a full 
derivation of the formulas used in the code.
