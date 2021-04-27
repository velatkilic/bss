# -*- coding: utf-8 -*-
"""
Basic simulator for blind source separation
"""

import numpy as np
import matplotlib.pyplot as plt

cs = 343       # speed of sound in air m/s
fs = 16e3      # sampling rate in Hz
T  = 0.1       # total observation time in s
N  = int(fs*T) # total number of points
t  = np.linspace(0,N-1,N) # time array

# spherical wave
def sWave(x,y,z,f):
    r = np.sqrt(x**2 + y**2 + z**2) # distance in m
    wavelength = f/cs;
    k = 2*np.pi/wavelength
    return np.cos(k*r - 2*np.pi*f*t/fs)/r

# convert spherical coordinates to cartesian
def s2c(r,theta,phi):
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    return x,y,z

# same wave recorded by a 4 microphone array
r     = 10
theta = np.pi/4
phi   = np.pi/3

x,y,z = s2c(r,theta,phi)

f = 0.01*fs

m1 = sWave(x+.05,y,z,f)
m2 = sWave(x,y+.05,z,f)
m3 = sWave(x-.05,y,z,f)
m4 = sWave(x,y-.05,z,f)

plt.figure()
plt.plot(t,m1,t,m2,t,m3,t,m4)


# Experiment 1: Separate two different sources

# Wave 1
r1     = 10
theta1 = np.pi/4
phi1   = np.pi/3

x1,y1,z1 = s2c(r1,theta1,phi1)

f1 = 0.01*fs

# Wave 2
r2     = 15
theta2 = np.pi/3
phi2   = np.pi/5

x2,y2,z2 = s2c(r2,theta2,phi2)

f2 = 0.02*fs

# Microphone measurements
m10  = sWave(x1+.05,y1,z1,f1) + sWave(x2+.05,y2,z2,f2) 
m01  = sWave(x1,y1+.05,z1,f1) + sWave(x2,y2+.05,z2,f2)
mn10 = sWave(x1-.05,y1,z1,f1) + sWave(x2-.05,y2,z2,f2)
m0n1 = sWave(x1,y1-.05,z1,f1) + sWave(x2,y2-.05,z2,f2)

plt.figure()
plt.plot(t,m10,t,m01,t,mn10,t,m0n1)


# Calculate xi
avg = (m10 + m01 + mn10 + m0n1)/4
x00 = np.append((avg[0:N-1] - avg[1:N]),0)*1e3
x10 = 100*(m10 - mn10)/2. # 100 to convert from cm^-1 to m^-1
x01 = 100*(m01 - m0n1)/2.

plt.figure()
plt.plot(t,x10,t,x01)

# Perform ICA
from sklearn.decomposition import FastICA

X = np.zeros((N,3)) # 2 measurements, N samples
X[:,0] = x00
X[:,1] = x10
X[:,2] = x01

ica = FastICA(n_components=2, random_state=0)
S = ica.fit_transform(X)  # Reconstruct signals
A = ica.mixing_  # Get estimated mixing matrix

# Sources will be time derivatives of the waves so we integrate
from scipy import integrate
s1p = integrate.cumtrapz(S[:,0], t)
s1p = s1p/np.max(s1p)

s2p = integrate.cumtrapz(S[:,1], t)
s2p = s2p/np.max(s2p)

plt.figure()
plt.plot(t[0:N-1],s1p,t[0:N-1],s2p)

# Visualize sources
s1gt = sWave(x1,y1,z1,f1)
s1gt = s1gt/np.max(s1gt)

s2gt = sWave(x2,y2,z2,f2)
s2gt = s2gt/np.max(s2gt)

plt.figure()
plt.plot(t,s1gt,t,s2gt)


# From A matrix convert tau to r1 and r2
R = A/cs

# direction cosines
dCos1 = R[:,0]/R[0,0]
dCos2 = R[:,1]/R[0,1]

# azimuth angle
theta_pred1 = np.arctan2(dCos1[1],dCos1[2])
theta_pred2 = np.arctan2(dCos2[1],dCos2[2])

print("Predicted: " + str(theta_pred1) + " Actual: " + str(theta1))
print("Predicted: " + str(theta_pred2) + " Actual: " + str(theta2))







