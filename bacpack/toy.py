import numpy as np
import pandas as pd
from bacpack.bac import Sim_BN


csim=Sim_BN(3,10000) # creating an instance of the simulation class for 3 assets and 10000 time points
csim.frequencies(10,0.2) #setting the frequencies of trades according to the simulation model
csim.generate() #generating trading data
mt=np.dot(csim.comps.T,csim.comps) #integrated covarinace matrix at highest precision to compare with
csim.noise(0.1) #generating microstructre noise
rvar=csim.fvar() # realiazed variance for each of the assets
nf_diag=csim.TSRV(2) # TSRV
noise_var=(rvar-nf_diag)/2/csim.missing_points.sum(axis=0)# estimating noise variance
noise_var=np.maximum(noise_var,0)
na_diag=2*noise_var*csim.missing_points.sum(axis=0)               
m0_RC,b0_RC=csim.RCov() #returns RCov and implied beta                  
mweights=csim.meanw() #average weights
b0_RC_nf=b0_RC-na_diag*mweights                  
bHY=csim.beta_HYCov() # stock-ETF beta using Hayashi-Yoshida estimator
SBAC=m0_RC-csim.NBAC_Delta_NR(b0_RC_nf,bHY,na_HY) #BAC adjustment
print(np.linalg.norm(mt-m0_RC)**2,np.linalg.norm(mt-SBAC)**2) # squared errors of the pre-estimator and BAC
print(csim.HY()[0]) #HY estimate of the integrated covariance matrix
