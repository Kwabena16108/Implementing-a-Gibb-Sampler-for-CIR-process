# CIR-Bayesian-Estimation

The following code implements the CIR (Cox, Ingersoll and Ross, 1985) stochastic interest rate model via a Gibb Sampler 
algorithm and simulate bond prices in Python.

The process involves finding the full conditional posterior distribution of model parameters in closed form.
The Gibb Sampler algorithm is used to draw samples from the conditional posterior distribution.
Readers can refer to https://digitalcommons.sacredheart.edu/wcob_theses/22/ for my dissertation paper titled 
"A Bayesian Approach to Assessing the Risk Premium on Catastrophe Bond Derivatives at Issuance" for a full 
derivation of the formulas used in the code.
