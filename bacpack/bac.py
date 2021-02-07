
import numpy as np
from numba import jit, cuda, double, prange, njit
import pandas as pd
import math
from .highfreq import HFReturns, ffillz, fastvar


class HFR_ETF(HFReturns):
    def __init__(self,nvar=2, nper=100000):
        HFReturns.__init__(self,nvar=nvar,nper=nper)

    def generate_from_obsolete(self, prices, missingpoints,etf_col):
        self.numper=prices.shape[0]-1
        self.numvar=prices.shape[1]-1

        prices=prices.copy()

        cols=np.ones(prices.shape[1],dtype=bool)
        cols[prices.columns.get_loc(etf_col)]=False
        returns=(np.log(prices)-np.log(prices.iloc[0,:]))[1:]
        self.missing_points=np.ascontiguousarray(missingpoints.iloc[1:,cols].values)
        self.mp_etf=missingpoints.iloc[1:,prices.columns.get_loc(etf_col)].values

        cumrets=np.ascontiguousarray((returns.loc[:,cols]).values)

        self.wetf=prices.iloc[:,prices.columns.get_loc(etf_col)].values
        self.etf=np.diff(self.wetf)
        self.letf=np.diff(np.log(self.wetf))
        self.wetf=self.wetf[:-1]

        cumrets[np.logical_not(self.missing_points)]=np.nan
        ffillz(cumrets)
        self.acumrets=cumrets.copy()
        cumrets[1:,:]=np.diff(cumrets,axis=0)
        self.acomps=cumrets
        self.aweights=np.ascontiguousarray(prices.iloc[1:,cols].values).astype(np.float64)
        self.aweights*=self.amounts
        self.meanweights=np.array([self.aweights[:,c][self.missing_points[:,c]].sum()/
              self.missing_points[:,c].sum() for c in range(self.numvar)])
        self.acompmeans=self.acomps.mean(axis=0)
        self.meansqws=np.array([(self.aweights[:,c][self.missing_points[:,c]]**2).sum()/
              self.missing_points[:,c].sum() for c in range(self.numvar)])

    def generate_from(self, prices, mp, shnumb, nonequity, outstanding, etf_col):
        uprices=prices.fillna(method="ffill")
        uprices=uprices.fillna(method="bfill")
        self.amounts=np.array(shnumb)
        self.amounts/=outstanding
        uprices[etf_col]-=nonequity/outstanding
        self.generate_from_obsolete(uprices,mp,etf_col=etf_col)
        return 0


    def BAC_Delta_NR(self,b0,b,L_out=False,L_in=False,L=0):
        if not L_in:
            ### RBAC no adjsutment of the main diagonal 
            W=np.zeros((self.numvar,self.numvar**2))
            Q=np.zeros((self.numvar*self.numvar,self.numvar**2))
            mw=self.meanw()
            for i in range(self.numvar):
                W[i,i*self.numvar:(i+1)*self.numvar]=mw
            for i in range(self.numvar):
                for j in range(self.numvar):
                    if i!=j:
                        Q[i*self.numvar+j,j*self.numvar+i]=-0.5
                        Q[i*self.numvar+j,i*self.numvar+j]=0.5
            L=np.dot(np.eye(self.numvar**2)-Q,W.T)
            L=np.dot(L,np.linalg.inv(np.eye(self.numvar)*(
                       self.smeanw()).sum()
                        -np.linalg.multi_dot([W,Q,W.T])))
        if L_out:
            return np.dot(L,b0-b).reshape(self.numvar,self.numvar),L
        return np.dot(L,b0-b).reshape(self.numvar,self.numvar)

    def BAC_Delta(self,b0,b,L_out=False,L_in=False,L=0):
        if not L_in:
            ### RBAC no adjsutment of the main diagonal 
            W=np.zeros((self.numvar,self.numvar**2))
            Q=np.zeros((self.numvar*self.numvar,self.numvar**2))
            mw=self.meanw()
            for i in range(self.numvar):
                W[i,i*self.numvar:(i+1)*self.numvar]=mw
            for i in range(self.numvar):
                for j in range(self.numvar):
                    if i==j:
                        Q[i*self.numvar+i,i*self.numvar+i]=1
                    else:
                        Q[i*self.numvar+j,j*self.numvar+i]=-0.5
                        Q[i*self.numvar+j,i*self.numvar+j]=0.5
            L=np.dot(np.eye(self.numvar**2)-Q,W.T)
            L=np.dot(L,np.linalg.inv(np.eye(self.numvar)*(
                     self.smeanw()).sum()
                     -np.linalg.multi_dot([W,Q,W.T])))
        if L_out:
            return np.dot(L,b0-b).reshape(self.numvar,self.numvar),L
        return np.dot(L,b0-b).reshape(self.numvar,self.numvar)

    def NBAC_Delta(self,b0,b,noise,L_out=False,L_in=False,L=0):
        if not L_in:
            meanweights=self.meanw()
            meansqws=self.smeanw()
            ns=noise/self.missing_points.sum(axis=0)
            ### RBAC no adjsutment of the main diagonal 
            W=np.zeros((self.numvar,self.numvar**2))
            Q=np.zeros((self.numvar*self.numvar,self.numvar**2))
            for i in range(self.numvar):
                W[i,i*self.numvar:(i+1)*self.numvar]=meanweights*np.exp(ns/4)
            for i in range(self.numvar):
                for j in range(self.numvar):
                    if i==j:
                        Q[i*self.numvar+i,i*self.numvar+i]=1
                    else:
                        Q[i*self.numvar+j,j*self.numvar+i]=-0.5
                        Q[i*self.numvar+j,i*self.numvar+j]=0.5
            L=np.dot(np.eye(self.numvar**2)-Q,W.T)
            L=np.dot(L,np.linalg.inv(np.eye(self.numvar)*(meansqws*np.exp(ns/2)).sum()
                        -np.linalg.multi_dot([W,Q,W.T])))
        if L_out:
            return np.dot(L,b0-b).reshape(self.numvar,self.numvar)+np.diag(noise),L
        return np.dot(L,b0-b).reshape(self.numvar,self.numvar)+np.diag(noise)

    def NBAC_Delta_NR(self,b0,b,noise,L_out=False,L_in=False,L=0):
        if not L_in:
            meanweights=self.meanw()
            meansqws=self.smeanw()
            ns=noise/self.missing_points.sum(axis=0)
            ### RBAC no adjsutment of the main diagonal 
            W=np.zeros((self.numvar,self.numvar**2))
            Q=np.zeros((self.numvar*self.numvar,self.numvar**2))
            for i in range(self.numvar):
                W[i,i*self.numvar:(i+1)*self.numvar]=meanweights*np.exp(ns/4)
            for i in range(self.numvar):
                for j in range(self.numvar):
                    if i!=j:
                        Q[i*self.numvar+j,j*self.numvar+i]=-0.5
                        Q[i*self.numvar+j,i*self.numvar+j]=0.5
            L=np.dot(np.eye(self.numvar**2)-Q,W.T)
            L=np.dot(L,np.linalg.inv(np.eye(self.numvar)*(meansqws*np.exp(ns/2)).sum()
                        -np.linalg.multi_dot([W,Q,W.T])))
        if L_out:
            return np.dot(L,b0-b).reshape(self.numvar,self.numvar)+np.diag(noise),L
        return np.dot(L,b0-b).reshape(self.numvar,self.numvar)+np.diag(noise)


class Sim_BN(HFR_ETF):

## Barndorff-Nielsen(2011) setup simulation

 def __init__(self,nvar=2, nper=100000, mis_pnts=0.99,mis_pntse=0.99):
    HFR_ETF.__init__(self,nvar=nvar,nper=nper)
    self.numper=nper # the number of time periods to be generated (constant correlation)
    self.numvar=nvar # the number of index components
    self.missing_points_ratio=mis_pnts
    self.missing_points_ratioe=mis_pntse
    self.amounts=np.ones(self.numvar)
    self.mmu=np.zeros(self.numvar) # assuming zero-centered returns
    #self.inds=np.tril_indices(self.numvar,-1) # lower triangle indices
    self.mispntscale=np.linspace(self.missing_points_ratio,
                              self.missing_points_ratioe,self.numvar)

 def frequencies(self,factor=5,minf=0):
    fqs=np.exp(factor*np.linspace(0,1,self.numvar)-factor)*(1-minf)+minf
    self.mispntscale=1-fqs

 def generate(self,flag=False,flag_s=False, fexp=True, grid_preset=False, mp=0,
    mu=0.03,
    beta0=-5/16,
    beta1=1/8,
    alpha=-1/40,jumpsperiod=0, jumpmagnitude = 0.5, rho=-0.3):
    if flag_s:
        rho=np.random.rand(self.numvar)

    self.mp_etf=np.ones(self.numper,dtype=np.bool)

    dB=np.random.normal(0,1/np.sqrt(self.numper),(self.numper,self.numvar))
    vrho=np.empty((self.numper,self.numvar))
    vrho0=np.random.normal(0,np.sqrt(-1/2/alpha),self.numvar)#np.zeros(self.numvar)
    for i in range(self.numper):
        vrho[i,:]=vrho0+alpha*vrho0/self.numper+dB[i,:]
        vrho0=vrho[i,:]
    sigma=np.exp(beta0+beta1*vrho)
    if flag: sigma[1:,:]=sigma[:-1,:]
    dW=np.random.normal(0,1/np.sqrt(self.numper),(self.numper,1))
    dF=np.sqrt(1-rho**2)*sigma*dW

    Csigma=np.sqrt(1-rho**2)*sigma
    self.ssCov=(np.dot(Csigma.T,Csigma)+np.diag(np.diag(np.dot((rho*sigma).T,
                rho*sigma))))/self.numper

    self.comps=mu/self.numper+dF+rho*sigma*dB

    # simulating non-synchronous trading
    if fexp:
        self.missing_points=np.zeros_like(self.comps,dtype=np.bool)
        for k in range(self.numvar):
            tmp=np.floor(np.random.exponential(1/(1-self.mispntscale[k]),
              np.int32(np.floor(self.numper*(1-self.mispntscale[k])*1.3))).cumsum())
            tmp=np.int32(tmp[tmp<self.numper])
            self.missing_points[tmp,k]=True
    else:
        self.missing_points=np.random.rand(self.numper,
        self.numvar)>self.mispntscale
    if grid_preset:
        self.missing_points=np.logical_or(self.missing_points,mp)
    # building etf
    cumrets=np.cumsum(self.comps,axis=0)

    if(jumpsperiod>0):
        self.jfetf=np.diff((np.exp(cumrets)*self.amounts).sum(axis=1),
                           prepend=self.amounts.sum())
        jumpfrequency = jumpsperiod/self.numper
        meansigma = np.sqrt((self.comps**2).mean(axis=0))
        for l in range(self.numvar):
            jumpoccs=np.random.poisson(jumpfrequency,self.numper)
            jumptimes = np.arange(self.numper,dtype=np.int32)[jumpoccs>0]
            count=0
            for s in jumptimes:
                jumpsize=0
                for j in range(jumpoccs[s]):
                    jumpsize+=(jumpmagnitude*
                    np.sign(np.random.normal(0,1))*np.random.uniform(1,2)*
                    meansigma[l])
                    cumrets[s:,l]+=jumpsize
                    count+=1

    #self.cumrets=cumrets.copy()
    prices=np.exp(cumrets)*self.amounts
    priceetf=prices.sum(axis=1)
    #prices=(prices.T/priceetf).T
    #print(prices.mean(axis=0),priceetf.mean())
    self.letf=np.diff(np.log(priceetf),prepend=np.log(self.amounts.sum()))
    self.etf=(priceetf)-(self.amounts.sum())
    self.etf[1:]=np.diff(self.etf)
    self.wetf=priceetf
    #self.etf=np.zeros(self.numper)

    cumrets[np.logical_not(self.missing_points)]=np.nan
    ffillz(cumrets)
    #self.acumrets=cumrets.copy()
    cumrets[1:,:]=np.diff(cumrets,axis=0)
    self.acomps=cumrets
    self.weights=prices.copy()
    #self.weights[0,:]-=1
    #self.weights[1:,:]=np.diff(prices,axis=0)
    self.aweights=prices.copy()
    self.aweights[np.logical_not(self.missing_points)]=np.nan
    ffillz(self.aweights)


    self.meanweights=np.array([(prices[:,c][self.missing_points[:,c]]).sum()/
        (self.missing_points[:,c].sum())
           for c in range(self.numvar)])
    self.meansqws=np.array([(prices[self.missing_points[:,c],c]**2).sum()/
        (self.missing_points[:,c].sum())
           for c in range(self.numvar)])

 def gen_jmp(self, jumpsperiod=0, jumpmagnitude = 0.5):
        cumrets=np.cumsum(self.comps,axis=0)

        if(jumpsperiod>0):
            self.jfetf=np.diff((np.exp(cumrets)*self.amounts).sum(axis=1),
                               prepend=self.amounts.sum())
            jumpfrequency = jumpsperiod/self.numper
            meansigma = np.sqrt((self.comps**2).mean(axis=0))
            for l in range(self.numvar):
                jumpoccs=np.random.poisson(jumpfrequency,self.numper)
                jumptimes = np.arange(self.numper,dtype=np.int32)[jumpoccs>0]
                count=0
                for s in jumptimes:
                    jumpsize=0
                    for j in range(jumpoccs[s]):
                        jumpsize+=(jumpmagnitude*
                        np.sign(np.random.normal(0,1))*np.random.uniform(1,2)*
                        meansigma[l])
                        cumrets[s:,l]+=jumpsize
                        count+=1

        #self.cumrets=cumrets.copy()
        prices=np.exp(cumrets)*self.amounts
        priceetf=prices.sum(axis=1)
        #prices=(prices.T/priceetf).T
        #print(prices.mean(axis=0),priceetf.mean())
        self.letf=np.diff(np.log(priceetf),prepend=np.log(self.amounts.sum()))
        self.etf=(priceetf)-(self.amounts.sum())
        self.etf[1:]=np.diff(self.etf)
        self.wetf=priceetf
        #self.etf=np.zeros(self.numper)

        cumrets[np.logical_not(self.missing_points)]=np.nan
        ffillz(cumrets)
        #self.acumrets=cumrets.copy()
        cumrets[1:,:]=np.diff(cumrets,axis=0)
        self.acomps=cumrets
        self.weights=prices.copy()
        #self.weights[0,:]-=1
        #self.weights[1:,:]=np.diff(prices,axis=0)
        self.aweights=prices.copy()
        self.aweights[np.logical_not(self.missing_points)]=np.nan
        ffillz(self.aweights)


        self.meanweights=np.array([(prices[:,c][self.missing_points[:,c]]).sum()/
            (self.missing_points[:,c].sum())
               for c in range(self.numvar)])
        self.meansqws=np.array([(prices[self.missing_points[:,c],c]**2).sum()/
            (self.missing_points[:,c].sum())
               for c in range(self.numvar)])


 def noise(self,kappa):
        ### gererates noise and adjusts prices and returns

        asvar=(self.comps**2).sum(axis=0)*kappa/self.missing_points.sum(axis=0)
        cumrets=np.random.normal(0,np.sqrt(asvar),(self.numper,self.numvar))

        cumrets[np.logical_not(self.missing_points)]=np.nan
        ffillz(cumrets)
        cumrets[1:,:]=np.diff(cumrets,axis=0)
        self.acomps+=cumrets
        etf_noise_std=np.sqrt((self.letf**2).sum()*kappa/self.mp_etf.sum())
        cumrets=np.random.normal(0,etf_noise_std,self.numper)
        cumrets[1:]=np.diff(cumrets)
        self.letf+=cumrets
        self.etf=np.diff(np.exp(self.letf.cumsum()+np.log(self.amounts.sum())),prepend=self.amounts.sum())

