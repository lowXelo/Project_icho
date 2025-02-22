import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
import os
import re
import time
import scipy
from scipy import constants
from scipy import io
from skimage import transform
from sklearn import metrics
from scipy import interpolate
from collections import Counter
from functools import reduce
import json
import pickle
import csv
import pandas as pd

def replacenth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n - 1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted)
    newString = before + after
    return newString

def create_refl_file(direfl, val):
    f = open(direfl+"sprefl"+str(val)+".dat", "w")
    f.write("2\n")
    f.write("600.00\t"+str(val)+"\n")
    f.write("14000.00\t"+str(val)+"\n")
    f.close()
          
def replace_line(filename, *args):
    # Open the file and read all the lines from the file into a list 'lines'
    with open(filename) as file:
        lines = file.readlines()
        # if the line number is in the file, we can replace it successfully
        for ar in range(0,len(args)-1,2):
            #print(ar)
            line_number=args[ar]
            val=args[ar+1]
            newval=str(val)
            if line_number>=50 and line_number<60:
                gazcoord=line_number-50
                line_number=51
            if line_number>=60 and line_number<70:
                gazcoord=line_number-60
                line_number=60
            
            if (line_number <= len(lines)):
                oldtext=lines[line_number]
                if line_number in [9, 12]: #albedo, angle
                    oldval=re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",oldtext)
                    oldtext=oldtext.replace(oldval[-1],newval)
                elif line_number==34: #jacobian
                    #to do
                    oldval=re.findall("YES|NO", oldtext)
                    oldtext=oldtext.replace(oldval[-1],newval)
                elif line_number==60 or line_number==51: #CO2, H20, CH4 jacob
                     toto = oldtext.split(",")
                     if gazcoord==0:
                         toto[gazcoord]="    " + newval + " "
                     else:
                         toto[gazcoord]=" " + newval + " "
                     oldtext=",".join(toto)
                elif line_number in [16, 17]: #solar zenith, azimuth
                    oldval=re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",oldtext)
                    oldtext=oldtext.replace(oldval[-1],newval)
                lines[line_number] = oldtext


            # solution
            # toto = oldtext.split(",")
            # toto[gazcoord]=str(newval)
            # oldtext=",".join(toto)
            
    # Open the file in 'writing mode' using the 2nd argument "w", this means 
    # that the file will be made blank, and any new text we write to the file 
    # will become the new file contents.
    with open(filename, "w") as file:
        # Loop through the list of lines, write each of them to the file
        for line in lines:
            file.write(line)

    
def from3to2(img, Nx, Ny):
  Ni=int(img.shape[0])
  Nj=int(img.shape[1])
  Nthumb=int(img.shape[2])
  if Nthumb != Nx*Ny:
      print('Warning, wrong number of x/y thumbnails')
    
  imstack=np.zeros((Nx*Ni,Ny*Nj))
  k=0
  
  for i in np.arange(Nx):
    for j in np.arange(Ny):
      imstack[i*Ni:(i+1)*Ni,j*Nj:(j+1)*Nj]=img[:,:,k]
      k=k+1
  return imstack

def from2to3(img, Nx, Ny):
  Ni=int(img.shape[0]/Nx)
  Nj=int(img.shape[1]/Ny)
  Nthumb=Ni*Nj
  imstack=np.zeros((Nx,Ny,Nthumb))
  k=0
  for j in np.arange(Nj):
    for i in np.arange(Ni):
      imstack[:,:,k]=img[i*Nx:(i+1)*Nx,j*Ny:(j+1)*Ny]
      k=k+1
  return imstack

def find_outliers_1D(data, pclow=None, pchigh=None, thresh=None):
    data=np.ravel(data)
    if pclow is None:
        pclow=25
    if pchigh is None:
        pchigh=75
    if thresh is None:
        thresh=1.5
    q1 = np.percentile(data, pclow)
    q3 = np.percentile(data, pchigh)
    iqr = q3 - q1
    lower_fence = q1 - thresh * iqr
    upper_fence = q3 + thresh * iqr
    outliers = np.where((data < lower_fence) | (data > upper_fence))
    no_outliers = np.where((data >= lower_fence) & (data <= upper_fence))
    return outliers[0], no_outliers[0]


def find_outliers_2D(data, pclow=None, pchigh=None, thresh=None):
    if pclow is None:
        pclow=25
    if pchigh is None:
        pchigh=75
    if thresh is None:
        thresh=1.5
    q1 = np.percentile(data.flatten(), pclow)
    q3 = np.percentile(data.flatten(), pchigh)
    iqr = q3 - q1
    lower_fence = q1 - thresh * iqr
    upper_fence = q3 + thresh * iqr
    outliers = np.argwhere((data < lower_fence) | (data > upper_fence))
    no_outliers = np.argwhere((data >= lower_fence) & (data <= upper_fence))
    return outliers, no_outliers

def joint_entropies(data, nbins=None):
    n_variables = data.shape[-1]
    n_samples = data.shape[0]
    if nbins == None:
        nbins = int((n_samples/5)**.5)
    histograms2d = np.zeros((n_variables, n_variables, nbins, nbins))
    for i in range(n_variables):
        for j in range(n_variables):
            histograms2d[i,j] = np.histogram2d(data[:,i], data[:,j], bins=nbins)[0]
    probs = histograms2d / len(data) + 1e-100
    joint_entropies = -(probs * np.log2(probs)).sum((2,3))
    return joint_entropies

def mutual_info_matrix(data, nbins=None, normalized=True):
    n_variables = data.shape[-1]
    j_entropies = joint_entropies(data, nbins)
    entropies = j_entropies.diagonal()
    entropies_tile = np.tile(entropies, (n_variables, 1))
    sum_entropies = entropies_tile + entropies_tile.T
    mi_matrix = sum_entropies - j_entropies
    if normalized:
        mi_matrix = mi_matrix * 2 / sum_entropies    
    return mi_matrix

def perf_registration(imthumb, cref):
    [Nx,Ny,Nthumb]=imthumb.shape
    imref=imthumb[:,:,cref]

    cc=np.zeros((Nthumb))
    mi=np.zeros((Nthumb))
    
    
    for ii in np.arange(Nthumb):
        cc[ii]=np.corrcoef(imthumb[:,:,ii].flatten(), imref.flatten())[0,1]
        mi[ii]=mutual_info_matrix(np.stack([imthumb[:,:,ii].flatten(),imref.flatten()],axis=1))[0,1]
        
    return cc, mi




# input modules and functions ###########################################
with open("C:\\Users\\valen\\Desktop\\ICHO\\RESSOURCES\\import_modules.py") as mymodule:
    exec(mymodule.read())

with open("C:\\Users\\valen\\Desktop\\ICHO\\RESSOURCES\\my_pythfunc.py") as mypythfile:
    exec(mypythfile.read())

from alive_progress import alive_bar
from skimage.registration import optical_flow_ilk, optical_flow_tvl1
import cv2 as cv
from skimage.transform import warp

#read file

Nobs=4800
mycref=33
Nthumb=80

iim=2000
file_unregistered = "C:\\Users\\valen\\Desktop\\ICHO\\L1a_images_cube2000.fits"
#file_unregistered="L1a_images_cube"+str(iim)+".fits"
hdul = fits.open(file_unregistered)
unregim=hdul[0].data.astype(float)
unregim=np.transpose(unregim,(1,2,0))

for tt in np.arange(Nthumb):
    medfi=scipy.signal.medfilt2d(unregim[:,:,tt])
    outl,noout=find_outliers_2D(unregim[:,:,tt],pclow=10, pchigh=90)
    for oo in np.arange(len(outl)):
        unregim[outl[oo][0],outl[oo][1],tt]=medfi[outl[oo][0],outl[oo][1]]


plt.ion()
plt.figure()
plt.imshow(from3to2(unregim,8,10))

#perf_registration(unregim,33)

plt.figure()
plt.imshow(unregim[:,:,0])

plt.figure()
plt.imshow(unregim[:,:,1])

image0 = unregim[:,:,0]
image1 = unregim[:,:,1]

v,u = optical_flow_ilk(unregim[:,:,0],unregim[:,:,1])

plt.figure()
plt.plot(v)

nr, nc = image0.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

image1_warp = warp(image1, np.array([row_coords + v, col_coords + u]), mode='edge')


#plt.imshow(unregim[:,:,79])

reg_im = np.zeros((nr, nc, 3))
reg_im[..., 0] = image1_warp
reg_im[..., 1] = image0
reg_im[..., 2] = image0



plt.figure()
plt.imshow(image1_warp)


cc1 = np.corrcoef(image1.flatten(), image0.flatten())[0,1]
cc2 = np.corrcoef(image1_warp.flatten(), image0.flatten())[0,1]

print(cc1,cc2)





