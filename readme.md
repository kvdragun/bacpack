
# Beta-Adjusted Covariance Estimation

## Introduction

Exchange Traded Funds (ETFs) are often more liquid than many of their component stocks.  We exploit this feature to improve pre-estimators of the integrated covariance of the stocks included in the ETF. The proposed  Beta Adjusted Covariance (BAC) equals the pre-estimator plus a minimal adjustment matrix such that the covariance-implied stock-ETF beta equals a target beta. 

## The Authors
The package is based upon the reaserch and the results of Beta Adjusted Covariance Estimation paper by Kris Boudt, Kirill Dragun, Orimar Sauri and Steven Vanduffel.
The authors of the code are Kirill Dragun, Kris Boudt and Emil Sjørup.

## Package

### Installation

```console
$ pip install bacpack
```
### submodules and classes

- highreq: high frequency trading data representation and processing functionality. Class HF_Returns defines integrated covariance matrix estimators as member functions :
    - bacpack.highfreq.HY() : computes Hayashi-Yoshida estimator;
    - bacpack.highfreq.RCov() : computes realized covariance estimator;
    - TSCov() : computes two time scale estimator.
- bac: defines ETF related data structure HFR_ETF and Beta Adjusted Covariance Estimation functions:
  - bacpack.bac.HFR_ETF: extends HFReturns, adds ETF and BAC related functionality:
    - bacpack.bac.HFR_ETF.generate_from(aprices, mp, shnumb, nonequity, outstanding, etf_col):
        - aprices - Pandas DataFrame with prices for assets in columns, each row for unique tick time;
        - mp - Boolean array with True for the element traded at specified moment;
        - shnumb - an array with amounts of components/assets included in the ETF;
        - nonequity - aggregated value of the additional components (like cash, money-market funds, bonds, etc.), prices for which are not included in the aprices array;
        - outstanding - ETF shares outstanding;
        - etf_col - column name for the ETF.
    - bacpack.bac.BAC_Delta_NR(b0, b), implements non-restricted BAC:
        - b0 - implied pairwise beta of the pre-estimator;
        - b - target stock-ETF beta.
    - bacpack.bac.NBAC_Delta_NR(b0, b, noise), implements non-restricted NBAC, adjusted for noise effect :
        - b0 - implied pairwise beta of the pre-estimator;
        - b - target stock-ETF beta;
        - noise, cumulative noise variance of diagonal elements.
  -  bacpack.bac.Sim_BN: extends HFR_ETF adding data generations functionality for the simulation purposes:
    - bacpack.bac.Sim_BN.generate(): generates data;
    - bacpack.bac.Sim_BN.gen_jmp(self, jumps_per_period, jump_magnitude): generates jumps 
    - bacpack.bac.Sim_BN.noise(kappa): generates and adds microstructure noise with size given by kappa parameter.
    
### toy example

```
import numpy as np
from bacpack.bac import Sim_BN

csim=Sim_BN(3,10000) # creating an instance of the simulation class for 3 assets and 10000 time points
csim.frequencies(10,0.2) #setting the frequencies of trades according to the simulation model
csim.generate() #generating trading data
### by now csim object of Sim_BN class contains fields with a simulated dataset 

mt=np.dot(csim.comps.T,csim.comps) #integrated covarinace matrix at highest precision to compare with
csim.noise(0.1) #generating microstructre noise
na_diag=csim.noise_var() # estimating noise variance           
m0_RC,b0_RC=csim.RCov() #returns RCov and implied beta                  
mweights=csim.meanw() #average weights
b0_RC_nf=b0_RC-na_diag*mweights   #beta adjusted for noise variance               
bHY=csim.beta_HY() # stock-ETF beta using Hayashi-Yoshida estimator
SBAC=m0_RC-csim.NBAC_Delta_NR(b0_RC_nf,bHY,na_diag) #BAC adjustment
print(csim.HY()[0]) #HY estimate of the integrated covariance matrix       
print(np.linalg.norm(mt-m0_RC)**2,np.linalg.norm(mt-SBAC)**2) # squared errors of the pre-estimator and BAC
```

