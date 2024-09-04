# -*- coding: utf-8 -*-
"""
Test FFT methods
"""

import os
cd=os.getcwd()
import sys
import numpy as np
import utils as utl
from matplotlib import pyplot as plt
from scipy.fft import fft, fftshift, fftfreq
from scipy import signal
from scipy.signal import lombscargle

#%% Inputs
N=1000
dt=1

#%% Functions
def phase(c):
    c[np.abs(c)<10**-10]=np.nan
    return np.angle(c)

#%% Initialization
xn=np.random.rand(N)

time=np.arange(N)*dt

n=np.arange(N)
k=np.arange(N)

k_mod=k.copy()
k_mod[k_mod>=N/2]=N/2-k_mod[k_mod>=N/2][::-1]-1

NN,KK=np.meshgrid(n,k)
DFM=np.exp(-1j*KK*NN*2*np.pi/N)

#%% Main
omega_k=2*np.pi/N*k_mod/dt
Xk=np.matmul(DFM,xn-np.mean(xn))


Xk2=fft(xn-np.mean(xn))
omega_k2 = 2*np.pi*(fftfreq(N,d=dt))

Bk_ds=np.real(Xk*np.conj(Xk))

if N/2==int(N/2):
    psd=np.concatenate([[Bk_ds[0]],Bk_ds[1:int(N/2)]*2,[Bk_ds[int(N/2)]]])/N/dt
    f_psd=np.arange(int(N/2)+1)/N/dt
else:
    psd=np.concatenate([[Bk_ds[0]],Bk_ds[1:int(N/2)+1]*2])/N/dt
    f_psd=np.arange(int(N/2))/N/dt



f_psd2, psd2 = signal.periodogram(xn, fs=1/dt,  scaling='density')

f_psd3=f_psd2[1:]
psd3 = lombscargle(time, xn, f_psd3,normalize=False)


print(np.var(xn))
print(np.sum(psd)*np.diff(f_psd)[0])
print(np.sum(psd2)*np.diff(f_psd2)[0])
print(np.sum(psd3)*np.diff(f_psd3)[0])

#%% Plots

plt.figure()
plt.subplot(2,1,1)
plt.plot(omega_k,np.abs(Xk),'xk',label='Analytical')
plt.plot(omega_k2,np.abs(Xk2),'.r',label='scipy.fft')
plt.grid()

plt.subplot(2,1,2)
plt.plot(omega_k,phase(Xk),'xk')
plt.plot(omega_k2,phase(Xk2),'.r')
plt.yticks(np.array([-1,0,1])*np.pi,labels=[r'$-\pi$',r'$0$',r'$\pi$'])
plt.grid()

plt.figure()
plt.plot(f_psd,psd,'xk',label='Analytical')
plt.plot(f_psd2,psd2,'.r',label='scipy.periodogram')
plt.plot(f_psd3,psd3,'.b',label='scipy.lombscargle')
plt.grid()
