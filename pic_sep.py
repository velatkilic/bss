# -*- coding: utf-8 -*-
"""
Use HJ Algorithm on Picture Mixtures
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# Import images
cwd   = os.getcwd()

lenna = plt.imread(cwd+"\\c\\lenna.bmp")/255.
baboo = plt.imread(cwd+"\\c\\baboon.bmp")/255.

os.mkdir(cwd+"\\test")

def nrm(x):
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn)/ (mx - mn)

# Mix images
d = 0.1
A = np.array([[0.5+d, 0.5-d],
              [0.5-d, 0.5+d]])

m1 = A[0,0]*lenna + A[0,1]*baboo
m2 = A[1,0]*lenna + A[1,1]*baboo

# subtract mean
m1 = m1 - np.mean(m1)
m2 = m2 - np.mean(m2)

# Demix with HJ algorithm
c12, c21 = 0.1, 0.2                         # demixing coefficients
sp1      = np.zeros((512,512))              # predicted/separated signal 1
sp2      = np.zeros((512,512))              # predicted/separated signal 2
f        = lambda x: x**3                   # nonlinear function 1
g        = lambda x: np.arctan(x)           # nonlinear function 2
lr       = 1e0                             # learning rate

N = 10000
cnt = 0
for i in range(N):    
    # Separate signals based on demixing matrix C
    sp1 = (m1 - c12*m2)/(1 - c12*c21)
    sp2 = (m2 - c21*m1)/(1 - c12*c21)
    
    # Update demixing matrix C
    c12 = c12 + lr*np.mean(f(sp1)*g(sp2))
    c21 = c21 + lr*np.mean(f(sp2)*g(sp1))
    
    if (c12*c21>1):
        print("Error: c12*c21 must be smaller than 1  for stable recovery")
    
    if i % 10 == 0:
        cnt+=1
        plt.imsave(cwd+"\\test\\frame1_"+str(cnt).zfill(3)+".png", nrm(sp1))
        plt.imsave(cwd+"\\test\\frame2_"+str(cnt).zfill(3)+".png", nrm(sp2))


fig, ax = plt.subplots(3,2)
ax[0,0].imshow(nrm(lenna))
ax[0,1].imshow(nrm(baboo))
ax[1,0].imshow(nrm(m2))
ax[1,1].imshow(nrm(m1))
ax[2,0].imshow(nrm(sp1))
ax[2,1].imshow(nrm(sp2))
for i in range(3):
    for j in range(2):
        ax[i,j].set_xticklabels([])
        ax[i,j].set_yticklabels([])
plt.subplots_adjust(wspace=0, hspace=0)
