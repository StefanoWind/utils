# -*- coding: utf-8 -*-
"""
Test feature selection through random forest
"""

import os
cd=os.path.dirname(__file__)

import utils as utl
import numpy as np
import utils
from matplotlib import pyplot as plt
import warnings
import matplotlib

warnings.filterwarnings('ignore')
plt.close('all')

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Input
M=5#number of features
N=2000#number of samples

#%% Initialization
X=np.random.normal(0,1,(N,M))
Y=X*0

#%% Main

#build output
Y[:,0]=X[:,0]*0
Y[:,1]=X[:,1]
Y[:,2]=-X[:,2]
Y[:,3]=X[:,3]**2
Y[:,4]=np.sin(X[:,4]/np.ptp(X[:,4])*6*np.pi)

Y_std=np.tile(np.std(Y,axis=0),(N,1))
y=np.sum(Y/(Y_std+10**-10),axis=1)

#calculates correlation coefficient
corr=[]
for i in range(M):
    corr=np.append(corr,np.corrcoef(X[:,i],y)[0][1])
    
importance,importance_std,y_pred,test_mae,train_mae,best_params=utl.RF_feature_selector(X,y)
    
#%% Plots
plt.close('all')
fig=plt.figure(figsize=(18,3.5))
for m in range(M):
    plt.subplot(1,M,m+1)
    plt.plot(X[:,m],y,'.k',alpha=0.5)
    plt.plot(X[:,m],Y[:,m],'.b',markersize=2)
    plt.grid()
    plt.xlabel(r'$x_'+str(m)+'$')
    plt.ylabel('$y$')
utl.remove_labels(fig)
plt.tight_layout()

plt.figure()
plt.bar(np.arange(M)*2-0.25,importance,yerr=importance_std,color=(0,0,0,0.5),capsize=5,linewidth=2,width=0.5,label='Random Forest')
plt.bar(np.arange(M)*2+0.25,corr,color=(0,0,1,0.5),width=0.5,label='Correlation')
plt.legend()
plt.grid()

plt.figure()
utl.plot_lin_fit(y, y_pred)
plt.xlabel('$y(x)$')
plt.ylabel(r'$f_{RN}(x)$')


