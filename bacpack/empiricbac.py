import sys
import os

import numpy as np
import pandas as pd
import math
from numba import cuda
from .bac import HFR_ETF
from .highfreq import ffillz

class Empirics(HFR_ETF):
    def __init__(self,nvar=2, nper=100000, mis_pnts=0.99,mis_pntse=0.99):
        HFR_ETF(nvar=nvar,nper=nper)
        self.numper=nper # the number of time periods to be generated (constant correlation)
        self.numvar=nvar # the number of index components
        self.missing_points_ratio=mis_pnts
        self.missing_points_ratioe=mis_pntse
        self.amounts=np.ones(self.numvar)
        self.mmu=np.zeros(self.numvar) # assuming zero-centered returns
        #self.inds=np.tril_indices(self.numvar,-1) # lower triangle indices
        self.mispntscale=np.linspace(self.missing_points_ratio,
                                  self.missing_points_ratioe,self.numvar)

 
    def gen_from_db(self,fhdf5,prekey,tickers,shnumb,etf,beg,end,
                 nonequity,outstanding,LFRC=False,jfilt=0,difret=False,
                 exmax=0,):
        
        df_set=[]
        with pd.HDFStore(fhdf5) as hdf:
            for t in tickers:
                #print("loading:", t)
                if exmax>0:
                    tmp=hdf.select(prekey+t,where=["index>=beg","index<end"],
                            columns=["PRICE","EX"])
                    mex=tmp.groupby("EX").EX.count().sort_values().index[-exmax:]
                    tmp=tmp[tmp.EX.isin(mex)]
                    #print("Exchage filter: ",rd,tmp.shape[0])
                    tmp=tmp[["PRICE"]].rename(columns={"PRICE":t})
                else:     
                    tmp=hdf.select(prekey+t,where=["index>=beg","index<end"],
                            columns=["PRICE"]).rename(columns={"PRICE":t})
                #tmp=tmp[~tmp.index.duplicated(keep='first')]
                if jfilt>0:
                    sr=tmp.diff()**2
                    av=sr.mean()
                    tmp=tmp[sr<av*jfilt]
                tmp=tmp.groupby(tmp.index).mean()
                df_set.append(tmp)
            if exmax>0:
                    tmp=hdf.select(prekey+etf,where=["index>=beg","index<end"],
                            columns=["PRICE","EX"])
                    mex=tmp.groupby("EX").EX.count().sort_values().index[-exmax:]
                    tmp=tmp[tmp.EX.isin(mex)]
                    tmp=tmp[["PRICE"]].rename(columns={"PRICE":etf})
            else:
                tmp=hdf.select(prekey+etf,where=["index>=beg","index<end"],
                            columns=["PRICE"]).rename(columns={"PRICE":etf})
            #tmp=tmp[~tmp.index.duplicated(keep='first')]
            if jfilt>0:
                    sr=tmp.diff()**2
                    av=sr.mean()
                    tmp=tmp[sr<av*jfilt]
            tmp=tmp.groupby(tmp.index).mean()
            df_set.append(tmp)
        sumdf=0
        for ind,df in enumerate(df_set):
            if df.shape[0]==0:
                print("not loaded :",tickers[ind])
                return -1
            else:
                df.diff
                sumdf+=df.shape[0]
        prices=pd.concat(df_set,axis=1)
        print("trades loaded: ",sumdf," concat: ",prices.shape[0]) 
        uprices=prices.fillna(method="ffill")
        uprices=uprices.fillna(method="bfill")
        self.amounts=np.array(shnumb)
        self.so=(uprices.iloc[:,:-1]*self.amounts).sum(axis=1).mean()+nonequity
        self.so/=uprices[etf].mean()
        self.amounts/=outstanding
        if LFRC:
            print("minute resampling")
            mprices=uprices.drop(etf,axis=1).resample("T").last()
            logrets=np.log(mprices).diff(axis=0).values[1:,:]
            self.lfcov=np.dot(logrets.T,logrets)
            if np.isnan(self.lfcov).sum()>0:
                print(logrets)
            print("done")
        if difret:
            print("shrout dif:",outstanding-self.so)
            self.NAVdiff=(uprices.iloc[:,:-1]*self.amounts).sum(axis=1)-uprices[etf]
            self.NAVdiff+=nonequity/outstanding
            self.NAVdiff/=uprices[etf]
        else:
            uprices[etf]-=nonequity/outstanding
            print("generating matrix")
            self.generate_from(uprices,pd.notnull(prices),etf_col=etf)
        return 0