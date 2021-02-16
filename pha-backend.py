#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: pha-backend.py
#------------------------------------------------------------------------------
# Version 0.3
# 15 February, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
#import nc_time_axis
#import cftime
import scipy
import scipy.stats as stats    
from sklearn.preprocessing import StandardScaler
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
import cmocean
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
# OS libraries:
import os, glob
import os.path
from pathlib import Path
import sys
import subprocess
from subprocess import Popen
import time

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 20
nheader = 0
pha_startyear = 1895
pha_endyear = 2015
flag_absolute = True

dir_cwd = '/home/patternizer/Documents/REPOS/pha-backend/'
dir_raw = 'ICELAND/data/benchmark/world1/monthly/raw/'
#dir_pha = 'ICELAND/data/benchmark/world1/monthly/FLs.r00/'
dir_pha = 'ICELAND/data/benchmark/world1/monthly/WMs.r00/'
dir_stnlist = 'ICELAND/data/benchmark/world1/meta/world1_stnlist.tavg'
raw_files = sorted(glob.glob(dir_cwd+dir_raw+'*.raw.tavg'))
#pha_files = sorted(glob.glob(dir_cwd+dir_pha+'*.FLs.r00.tavg'))
pha_files = sorted(glob.glob(dir_cwd+dir_pha+'*.WMs.r00.tavg'))
stnlist = glob.glob(dir_cwd+dir_stnlist)[0]

#------------------------------------------------------------------------------
# LOAD: stnlist
#------------------------------------------------------------------------------

f = open(stnlist)
lines = f.readlines()
stationcodes = []
for i in range(nheader,len(lines)):
    words = lines[i].split()    
    stationcode = words[0]
    stationcodes.append(stationcode)        
f.close()    

n = len(stationcodes)

#------------------------------------------------------------------------------
# LOAD: PHA station adjusted absolute temperatures (Tx100) in GHCNm-v3 format
#------------------------------------------------------------------------------

df_pha = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])

for k in range(n):

    f = open(pha_files[k])
    lines = f.readlines()
    dates = []
    obs = []
    for i in range(nheader,len(lines)):
        words = lines[i].split()    
        date = int(words[0][-4:])
        val = (12)*[None]
        for j in range(len(val)):                            
            try: val[j] = int(words[j+1].rstrip('E'))
            except:                    
                pass
        dates.append(date)        
        obs.append(val) 
    f.close()    
    dates = np.array(dates)
    obs = np.array(obs)

    # Create station dataframe

    df = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = dates
    df['stationcode'] = len(dates)*[stationcodes[k][-6:]]
    for j in range(12):        
        df[df.columns[j+2]] = [ obs[i][j] for i in range(len(obs)) ]

    # Replace monthly fill value -9999 with np.nan    

    for j in range(12):        
        df[df.columns[j+2]].replace(-9999, np.nan, inplace=True)

    # Apply /100 scale factor    

    for j in range(12):       
        df[df.columns[j+2]] = df[df.columns[j+2]]/100.0

    # Calculate station normals

    normals = df[(df['year']>1960) & (df['year']<1991)].mean()[2:]
    if flag_absolute == True:
       normals = np.zeros(12)
    
    # Calculate station anomalies

    for j in range(12):       
        df[df.columns[j+2]] = df[df.columns[j+2]] - normals[j]

    # Append station anomalies dataframe

    df_pha = df_pha.copy().append([df],ignore_index=True)

# Map onto 1895-2015 and calculate monthly mean

t = pd.date_range(start=str(pha_startyear), end=str(pha_endyear+1), freq='M')   
dg_pha = pd.DataFrame(columns=df_pha['stationcode'].unique(),index=t)   

for i in range(n):

    da = df_pha[df_pha['stationcode']==df_pha['stationcode'].unique()[i]]
    nyears = da.max()['year']-da.min()['year']+1
    startyear = da.min()['year']
    if nyears == len(da):
        ts_monthly = np.array(da.groupby('year').mean().iloc[:,0:]).ravel() 
        t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')   
    else:        
        ts_monthly = (nyears*12)*[np.nan]
        for j in range(len(da)):                
            ts_monthly[(da['year'].iloc[j]-startyear)*12:(da['year'].iloc[j]-startyear+1)*12-1] = np.array(da.groupby('year').mean().iloc[j,0:])
        t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')   
    db = pd.DataFrame({dg_pha.columns[i]:ts_monthly},index=t_monthly)
    dg_pha[dg_pha.columns[i]] = db

dg_pha['mean'] = dg_pha.mean(axis=1)

#------------------------------------------------------------------------------
# LOAD: raw station absolute temperatures (Tx100) in GHCNm-v3 format
#------------------------------------------------------------------------------

df_raw = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])

for k in range(n):

    f = open(raw_files[k])
    lines = f.readlines()
    dates = []
    obs = []
    for i in range(nheader,len(lines)):
        words = lines[i].split()    
        date = int(words[1])
        val = (12)*[None]
        for j in range(len(val)):                            
            try: val[j] = int(words[j+2])
            except:                    
                pass
        dates.append(date)        
        obs.append(val) 
    f.close()    
    dates = np.array(dates)
    obs = np.array(obs)

    # Create station dataframe

    df = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = dates
    df['stationcode'] = len(dates)*[stationcodes[k][-6:]]
    for j in range(12):        
        df[df.columns[j+2]] = [ obs[i][j] for i in range(len(obs)) ]

    # Replace monthly fill value -9999 with np.nan    

    for j in range(12):        
        df[df.columns[j+2]].replace(-9999, np.nan, inplace=True)

    # Apply /100 scale factor    

    for j in range(12):       
        df[df.columns[j+2]] = df[df.columns[j+2]]/100.0

    # Calculate station normals

    normals = df[(df['year']>1960) & (df['year']<1991)].mean()[2:]
    if flag_absolute == True:
       normals = np.zeros(12)
    
    # Calculate station anomalies

    for j in range(12):       
        df[df.columns[j+2]] = df[df.columns[j+2]] - normals[j]

    # Trim to df_pha station tmin and tmax

    tmin = df_pha[df_pha['stationcode']==df['stationcode'].unique()[0]].min()['year']
    tmax = df_pha[df_pha['stationcode']==df['stationcode'].unique()[0]].max()['year']
    dg = df[(df['year']>=tmin)&(df['year']<=tmax)]

    # Append station anomalies dataframe

    df_raw = df_raw.copy().append([dg],ignore_index=True)

# Map onto 1895-2015 and calculate monthly mean

t = pd.date_range(start=str(pha_startyear), end=str(pha_endyear+1), freq='M')
dg_raw = pd.DataFrame(columns=df_raw['stationcode'].unique(),index=t)   

for i in range(n):

    da = df_raw[df_raw['stationcode']==df_raw['stationcode'].unique()[i]]
    nyears = da.max()['year']-da.min()['year']+1
    startyear = da.min()['year']
    if nyears == len(da):
        ts_monthly = np.array(da.groupby('year').mean().iloc[:,0:]).ravel() 
        t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')   
    else:        
        ts_monthly = (nyears*12)*[np.nan]
        for j in range(len(da)):                
            ts_monthly[(da['year'].iloc[j]-startyear)*12:(da['year'].iloc[j]-startyear+1)*12-1] = np.array(da.groupby('year').mean().iloc[j,0:])
        t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')   
    db = pd.DataFrame({dg_raw.columns[i]:ts_monthly},index=t_monthly)
    dg_raw[dg_raw.columns[i]] = db

dg_raw['mean'] = dg_raw.mean(axis=1)

#------------------------------------------------------------------------------
# PLOT: raw v adjusted
#------------------------------------------------------------------------------

for i in range(n):

    raw_ts_yearly = dg_raw.iloc[:,i].rolling(12).mean()
    pha_ts_yearly = dg_pha.iloc[:,i].rolling(12).mean()

    figstr = stationcodes[i] + '-raw-v-pha.png'
    titlestr = stationcodes[i] + ': 12-month rolling means'
                 
    fig, ax = plt.subplots(figsize=(15,10))          
    plt.scatter(raw_ts_yearly.index, raw_ts_yearly, marker='s', facecolor='white', color='blue', alpha=1.0, zorder=1, label='raw')
    plt.scatter(pha_ts_yearly.index, pha_ts_yearly, marker='.', color='red', alpha=1.0, zorder=2, label='PHA-adjusted')
    plt.vlines(raw_ts_yearly.index, raw_ts_yearly, pha_ts_yearly, colors='lightgrey', zorder=0)
    ax.xaxis.grid(True, which='major')      
    ax.yaxis.grid(True, which='major')  
    plt.tick_params(labelsize=16)    
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('Year', fontsize=fontsize)
    if flag_absolute == True:
        plt.ylabel(r'Absolute temperature [°C]', fontsize=fontsize)
    else:
        plt.ylabel(r'Temperature anomaly (from 1961-1990) [°C]', fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close(fig)

#------------------------------------------------------------------------------
# PLOT: raw (mean) v adjusted (mean)
#------------------------------------------------------------------------------

mean_raw_ts_yearly = dg_raw['mean'].rolling(12).mean()
mean_pha_ts_yearly = dg_pha['mean'].rolling(12).mean()

figstr = 'raw-v-pha-means.png'
titlestr = 'Mean raw versus mean PHA-adjusted: 12-month rolling means'
                 
fig, ax = plt.subplots(figsize=(15,10))          
plt.scatter(mean_raw_ts_yearly.index, mean_raw_ts_yearly, marker='s', facecolor='white', color='blue', alpha=1.0, zorder=1, label='raw')
plt.scatter(mean_pha_ts_yearly.index, mean_pha_ts_yearly, marker='.', color='red', alpha=1.0, zorder=2, label='PHA-adjusted')
plt.vlines(mean_raw_ts_yearly.index, mean_raw_ts_yearly, mean_pha_ts_yearly, colors='lightgrey', zorder=0)
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
plt.tick_params(labelsize=16)    
plt.legend(loc='upper left', fontsize=12)
plt.xlabel('Year', fontsize=fontsize)
if flag_absolute == True:
    plt.ylabel(r'Absolute temperature [°C]', fontsize=fontsize)
else:
    plt.ylabel(r'Temperature anomaly (from 1961-1990) [°C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close(fig)

#------------------------------------------------------------------------------
# PLOT: raw (mean) - adjusted (mean)
#------------------------------------------------------------------------------

figstr = 'raw-v-pha-means-diff.png'
titlestr = 'Mean adjustments (raw-PHA): 12-month rolling means'
                 
fig, ax = plt.subplots(figsize=(15,10))          
plt.step(x=mean_raw_ts_yearly.index, y=mean_raw_ts_yearly-mean_pha_ts_yearly, marker='.', color='black', lw=0.5, alpha=1.0, zorder=0)
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
plt.tick_params(labelsize=16)    
plt.xlabel('Year', fontsize=fontsize)
if flag_absolute == True:
    plt.ylabel(r'Change in absolute temperature [°C]', fontsize=fontsize)
else:
    plt.ylabel(r'Change in temperature anomaly (from 1961-1990) [°C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close(fig)

#------------------------------------------------------------------------------
# PLOT: histogram of adjustments
#------------------------------------------------------------------------------

adj_all = []
for i in range(n):
    raw_ts_yearly = dg_raw.iloc[:,i].rolling(12).mean()
    pha_ts_yearly = dg_pha.iloc[:,i].rolling(12).mean()
    adj_ts_yearly = raw_ts_yearly-pha_ts_yearly
    adj_all.append(adj_ts_yearly)    
adj_all = np.array(adj_all).flatten()
mask = (np.isfinite(adj_all)) & (np.abs(adj_all)>0.001)
adjustments = adj_all[mask]

bins = 41
xmin = -1.0; xmax = 1.0
figstr = 'adjustment-histogram.png'
titlestr = 'Histogram of all adjustments (raw-PHA): 12-month rolling means'
                 
fig, ax = plt.subplots(figsize=(15,10))     
kde = stats.gaussian_kde(adjustments); x = np.linspace(xmin,xmax,1000)
plt.hist(adjustments, density=False, bins=bins, alpha=1.0, color='lightgrey')
ax1 = plt.gca()
ax2 = ax.twinx()
ax2.plot(x, kde(x), color='red', lw=3)
if flag_absolute == True:
    ax1.set_xlabel(r'Absolute temperature adjustment [°C]', fontsize=fontsize)
else:
    ax1.set_xlabel(r'Temperature anomaly (from 1961-1990) adjustment [°C]', fontsize=fontsize)
ax1.set_ylabel('Count', fontsize=fontsize)
ax1.set_xlim(xmin,xmax)
ax1.xaxis.grid(True, which='major')      
ax1.tick_params(labelsize=16)    
ax2.set_ylabel('Kernel density estimate (KDE)', fontsize=fontsize, color='red')
ax2.tick_params(labelsize=16, colors='red')    
ax2.spines['right'].set_color('red')
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close(fig)

#------------------------------------------------------------------------------
print('** END')

