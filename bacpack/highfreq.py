# -*- coding: utf-8 -*-
"""highfreq.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qtzD0qd34aijWA11HqWs8fQ_71AZNwaj
"""

import numpy as np
from numba import jit, cuda, double, prange, njit
import pandas as pd
import math

@jit
def ffillz(narr):
    # filling missing values
    for j in range(narr.shape[1]):
        for i in range(narr.shape[0]):
            if not np.isnan(narr[i,j]):
                fill_value=narr[i,j]
                break 
        for i in range(narr.shape[0]):
            if not np.isnan(narr[i,j]):
                fill_value=narr[i,j]
            else:
                narr[i,j]=fill_value
                               
@jit
def backfill(narr,mp):
    res=np.empty_like(narr)
    for j in range(narr.shape[1]):
        fl=0
        for i in range(narr.shape[0]-1,-1,-1):
            if mp[i,j]:
                fl=narr[i,j]
            res[i,j]=fl
    return res    

@jit
def TSRV_imp(x, m, K = 300, J = 1): 
        rets = x[m]
        n = len(rets)
        nbarK = (n - K + 1)/(K)
        nbarJ = (n - J + 1)/(J)
        adj = (1 - (nbarK/nbarJ))**(-1)
        sumJ=0
        sumK=0
        fJ=0
        fK=0
        retJ=0
        retK=0
        for i in range(n):
            retJ+=rets[i]
            retK+=rets[i]
            if i-fJ==J-1:
                sumJ+=retJ**2
                retJ-=rets[fJ]
                fJ+=1
            if i-fK==K-1:
                sumK+=retK**2
                retK-=rets[fK]
                fK+=1    
            
        return  adj * ((1/K) * sumK - ((nbarK/nbarJ) * (1/J) * sumJ))

@jit
def icov(ux,uy,mpx,mpy):
    ### RCOV refresh time: ux,uy-returns; mpx,mpy - trade occurance
    x=ux
    y=uy
    sum=0
    xf=False
    yf=False
    acumx=0
    acumy=0
    for i in range(x.shape[0]):
        if (mpx[i] and mpy[i]) or (mpx[i] and yf) or (mpy[i] and xf):
            sum+=(acumx+x[i])*(acumy+y[i])
            acumx=0
            acumy=0
            xf=False
            yf=False
            continue
        if mpx[i]:
            xf=True
            acumx+=x[i]
        if mpy[i]:
            yf=True    
            acumy+=y[i] 
    return sum                  

@jit
def hayow(ux,uy,mpx,mpy,wx):#Hayashi-Yoshida weighted 
    sum=0
    for i in range(ux.shape[0]):
        if mpx[i]:
            for j in range(i,ux.shape[0]):
                if mpy[j]:
                    sum+=ux[i]*wx[i]*uy[j]
                    break
            continue    
        if mpy[i]:
            for j in range(i,ux.shape[0]):
                if mpx[j]:
                    sum+=uy[i]*ux[j]*wx[j]
                    break
    return sum

@njit(parallel=True)
def fast_beta(acomps,etf,missing_points,mp_etf,b00):
    for i in prange(acomps.shape[1]):
       b00[i]=hayo(acomps[:,i],etf,
          missing_points[:,i],
          mp_etf)

@njit(parallel=True)
def fastBAC_beta_log(acomps,missing_points,aweights,letf,etf,wetf,mp_etf,
                     b00,aw,etf_noise):
    sqw=0
    logw=1/wetf**2
    for i in prange(acomps.shape[1]):
       b00[i]=hayow(acomps[:,i],etf,
          missing_points[:,i],
          mp_etf,logw*aweights[:,i])
       tw=(logw*aweights[:,i])[missing_points[:,i]]
       aw[i]=tw.mean()
       sqw+=(tw**2).mean()
    res=aw/sqw*((letf[mp_etf]**2).sum()-etf_noise-b00.sum())    
    return res   
    
@njit(parallel=True)
def fastBAC_logbeta_log(acomps,missing_points,aweights,letf,etf,wetf,mp_etf,
                     b00,aw,etf_noise):
    sqw=0
    for i in prange(acomps.shape[1]):
       b00[i]=hayow(acomps[:,i],letf,
          missing_points[:,i],mp_etf,aweights[:,i]/wetf)
       tw=(1/wetf*aweights[:,i])[missing_points[:,i]]
       aw[i]=tw.mean()
       sqw+=(tw**2).mean()
    res=aw/sqw*((letf[mp_etf]**2).sum()-etf_noise-b00.sum())    
    return res     

@njit(parallel=True)
def fastTSRV(acomps,missing_points,res,K=30,J=1):
    for i in prange(acomps.shape[1]):
        res[i]=TSRV_imp(acomps[:,i],missing_points[:,i],K,J)

@jit
def TSCov_mat(x, mx, res, rets, K=30, J = 1): #matrix version
    for k in range(x.shape[1]):
      for l in range(k+1):
       indK=0
       n = 0
       sumJ=0.
       sumK=0.
       fJ=0
       fK=0
       retJk=0.
       retJl=0.
       retKk=0.
       retKl=0.   
       fk=False
       fl=False
       currk=0.
       currl=0.
       for i in range(x.shape[0]):
         if mx[i,k]:
            fk=True
            currk+=x[i,k]
         if mx[i,l]:
            fl=True
            currl+=x[i,l]
         if fk and fl:
            fk=False
            fl=False
            retJk+=currk
            retJl+=currl
            retKk+=currk
            retKl+=currl
            rets[indK,0]=currk
            rets[indK,1]=currl
            currk=0.
            currl=0.
            if n-fJ==J-1:
              sumJ+=retJk*retJl
              retJk-=rets[fJ%K,0]
              retJl-=rets[fJ%K,1]
              fJ+=1
            if n-fK==K-1:
              sumK+=retKk*retKl
              retKk-=rets[fK%K,0]
              retKl-=rets[fK%K,1]
              fK+=1   
            indK=(indK+1)%K
            n+=1
       nbarK = (n - K + 1)/(K)
       nbarJ = (n - J + 1)/(J)
       adj = (1 - (nbarK/nbarJ))**(-1)
       res[k,l] =adj * ((1/K) * sumK - ((nbarK/nbarJ) * (1/J) * sumJ))
       res[l,k] =res[k,l]

@njit(parallel=True)
def fastRC2(acomps,mx,aw,res,beta0):
    for k in prange(acomps.shape[1]):
        for l in prange(acomps.shape[1]): 
            fk=False
            fl=False
            currk=0.
            currl=0. 
            wcurr=0.
            cnt=0
            sum=0. 
            beta=0.
            for i in range(acomps.shape[0]):
                wcurr+=aw[i,k]
                cnt+=1
                if mx[i,k]:
                  fk=True
                  currk+=acomps[i,k]
                if mx[i,l]:
                  fl=True
                  currl+=acomps[i,l]
                if fk and fl:
                  fk=False
                  fl=False
                  tmp=currk*currl
                  sum+=tmp
                  beta+=wcurr/cnt*tmp
                  currk=0.
                  currl=0.
                  wcurr=0.
                  cnt=0
            res[k,l]=sum
            beta0[k,l]=beta
                

@njit(parallel=True)
def fastvar(acomps,mx,res):
    for k in prange(acomps.shape[1]):
        sum=0. 
        for i in range(acomps.shape[0]):
            if mx[i,k]:
                sum+=acomps[i,k]**2
        res[k]=sum 

@njit(parallel=True)
def fastHY2(acomps,mx,aw, res, beta0):
    for k in prange(acomps.shape[1]):
        for l in prange(acomps.shape[1]):    
            sum=0. 
            beta=0.
            for i in range(acomps.shape[0]):
                if mx[i,k] or mx[i,l]:
                  tmp=acomps[i,k]*acomps[i,l]
                  sum+=tmp
                  beta+=tmp*aw[i,k]
            res[k,l]=sum
            beta0[k,l]=beta                
            
@njit(parallel=True)
def TSCov_par(x, mx, res, rets, K=30, J = 1):#parallel
    for k in prange(x.shape[1]):
      for l in prange(k+1):
       indK=0
       n = 0
       sumJ=0.
       sumK=0.
       fJ=0
       fK=0
       retJk=0.
       retJl=0.
       retKk=0.
       retKl=0.   
       fk=False
       fl=False
       currk=0.
       currl=0.
       for i in range(x.shape[0]):
         if mx[i,k]:
            fk=True
            currk+=x[i,k]
         if mx[i,l]:
            fl=True
            currl+=x[i,l]
         if fk and fl:
            fk=False
            fl=False
            retJk+=currk
            retJl+=currl
            retKk+=currk
            retKl+=currl
            rets[indK,0,k,l]=currk
            rets[indK,1,k,l]=currl
            currk=0.
            currl=0.
            if n-fJ==J-1:
              sumJ+=retJk*retJl
              retJk-=rets[fJ%K,0,k,l]
              retJl-=rets[fJ%K,1,k,l]
              fJ+=1
            if n-fK==K-1:
              sumK+=retKk*retKl
              retKk-=rets[fK%K,0,k,l]
              retKl-=rets[fK%K,1,k,l]
              fK+=1   
            indK=(indK+1)%K
            n+=1
       nbarK = (n - K + 1)/(K)
       nbarJ = (n - J + 1)/(J)
       adj = (1 - (nbarK/nbarJ))**(-1)
       res[k,l] =adj * ((1/K) * sumK - ((nbarK/nbarJ) * (1/J) * sumJ))
       res[l,k] =res[k,l]

@jit
def hayo(ux,uy,mpx,mpy):#Hayashi-Yoshida 
    sum=0
    for i in range(ux.shape[0]):
        if mpx[i]:
            for j in range(i,ux.shape[0]):
                if mpy[j]:
                    sum+=ux[i]*uy[j]
                    break
            continue    
        if mpy[i]:
            for j in range(i,ux.shape[0]):
                if mpx[j]:
                    sum+=uy[i]*ux[j]
                    break
    return sum

@njit(parallel=True)
def fMedRV(acomps,mx,res,buf):
    for k in prange(acomps.shape[1]):
        j=0
        sum=0
        for i in range(acomps.shape[0]):
            if mx[i,k]:
                buf[k,j%3]=np.abs(acomps[i,k])
                j+=1
            if j>=3:    
                sum+=np.median(buf[k,:])**2
        res[k]=np.pi/(6-4*np.sqrt(3)+np.pi)*j/(j-2)*sum

class HFReturns:
    def __init__(self,nvar=2, nper=100000, mis_pnts=0.99,mis_pntse=0.99):
        self.numper=nper # the number of time periods to be generated (constant correlation)
        self.numvar=nvar # the number of index components
        self.gpu=False

    def jumpsfree(self,var,var_noise,varetf,var_etfnoise,eta=25):
        last=np.zeros(self.numvar,dtype=np.int32)
        svar=var/self.numper
        setf=varetf/self.numper
        elast=0
        vmask=np.zeros(self.numper,dtype=np.bool)    
        for i in range(self.numper):
            hmask=self.acomps[i,self.missing_points[i,:]]**2>((svar*(i+1-last)+
                             2*var_noise)*eta)[self.missing_points[i,:]]
            if np.any(hmask):
                vmask[last[self.missing_points[i,:]][hmask].min():i+1]=True                
            last[self.missing_points[i,:]]=i+1
            if self.mp_etf[i] :
                if self.letf[i]**2>((setf*(i+1-elast)+
                             2*var_etfnoise)*eta):
                    vmask[elast:i+1]=True
                elast=i+1
        self.missing_points[vmask,:]=False
        self.mp_etf[vmask]=False
        return vmask.sum()

    def beta(self,b_low=0,e_up=0):#beta computation
        # beta stock-ETF using RCOV
        if e_up==0:
            e_up=self.numper
        res=np.zeros(self.numvar)
        for i in range(self.numvar):
            res[i]=icov(self.etf[b_low:e_up],self.acomps[b_low:e_up,i],
                     self.mp_etf[b_low:e_up],self.missing_points[b_low:e_up,i])
        return res
    
    def beta_HY(self):
        # beta stock-ETF using HY
        res=np.zeros(self.numvar)
        fast_beta(self.acomps,self.etf,self.missing_points,self.mp_etf,res)
        return res 
    
    def fBAC_beta_log(self,etf_noise=0):
        #VAB beta adjustment returns a correction for pre-estimated beta (fast/parallel version)
        b00=np.empty(self.numvar)
        aw=np.empty(self.numvar)
        return fastBAC_beta_log(self.acomps,self.missing_points,self.aweights,
                    self.letf,self.etf,self.wetf,self.mp_etf,b00,aw,etf_noise)
                    
    def fBAC_logbeta_log(self,etf_noise=0):
        #VAB beta adjustment for logbeta [X,Y]. Returns a correction for pre-estimated logbeta (fast/parallel version)
        b00=np.empty(self.numvar)
        aw=np.empty(self.numvar)
        res=fastBAC_logbeta_log(self.acomps,self.missing_points,self.aweights,
                    self.letf,self.etf,self.wetf,self.mp_etf,b00,aw,etf_noise)
        #print("BAC_beta b00:",b00.sum(),(self.letf[self.mp_etf]**2).sum()-etf_noise,
        #      TSRV_imp(self.letf,self.mp_etf,200,15))
        return res                
        
    def beta_oracle(self):
        #oracle beta
        return np.dot(self.comps.T,self.etf)
    
    def TSRV(self,K=30,J=1):
        res=np.empty(self.numvar)
        fastTSRV(self.acomps,self.missing_points,res,K,J)
        return res  
     
    def HYCov(self):
        # Hayashi-Yoshida integrated covariance estimator
        cov=np.empty((self.numvar,self.numvar))
        beta0=np.empty((self.numvar,self.numvar))
        ac=backfill(self.acomps,self.missing_points)
        aw=backfill(self.aweights,self.missing_points)
        fastHY2(ac,self.missing_points,aw,cov,beta0)   
        return cov,beta0.sum(axis=0)

    def RCov(self): 
        # realized covariance estimator
        cov=np.empty((self.numvar,self.numvar))
        beta0=np.empty((self.numvar,self.numvar))
        fastRC2(self.acomps,self.missing_points,self.aweights,cov,beta0)   
        return cov,beta0.sum(axis=0)
    
    def TSCov(self,K=30,J=1):
        res=np.empty((self.numvar,self.numvar))
        rets=np.empty((K,2,self.numvar,self.numvar))
        TSCov_par(self.acomps,self.missing_points,res,rets,K=K,J=J)
        return res

    def MedRV(self):
        res=np.empty(self.numvar)
        buf=np.empty((self.numvar,3))
        fMedRV(self.acomps,self.missing_points,res,buf)
        return res

    def meanw(self):
        return np.array([(self.aweights[:,c][self.missing_points[:,c]]).sum()/
            (self.missing_points[:,c].sum())
               for c in range(self.numvar)])

    def smeanw(self):
        return np.array([(self.aweights[:,c][self.missing_points[:,c]]**2).sum()/
            (self.missing_points[:,c].sum())
               for c in range(self.numvar)])    
    
    def wmeanw(self):
        prcetf=self.aweights.sum(axis=1)
        return np.array([(self.weights[:,c][self.missing_points[:,c]]
                          /prcetf[[self.missing_points[:,c]]]).sum()/
            (self.missing_points[:,c].sum())
               for c in range(self.numvar)])      
               
    def fvar(self):
        cov=np.empty(self.numvar)
        fastvar(self.acomps,self.missing_points,cov)
        return cov

    def noise_var(self, K=2):
        rvar=self.fvar() # realiazed variance for each of the assets
        nf_diag=self.TSRV(K) # TSRV
        return rvar-nf_diag # estimating noise variance          
