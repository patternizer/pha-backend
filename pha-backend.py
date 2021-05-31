#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: pha-backend.py
#------------------------------------------------------------------------------
# Version 0.5
# 25 May February, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
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
# Math libraries
import random

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
flag_absolute = True

dir_cwd = '/local/cqz20mbu/Documents/REPOS/pha-backend/'
#dir_cwd = '~/Documents/REPOS/pha-backend/'
#dir_raw = 'ICELAND/trausti-raw-normals/data/benchmark/world1/monthly/raw/'
#dir_pha = 'ICELAND/trausti-raw-normals/data/benchmark/world1/monthly/WMs.r00/'
#dir_stnlist = 'ICELAND/trausti-raw-normals/data/benchmark/world1/meta/world1_stnlist.tavg'

dir_raw = 'AUSTRALIA/Adjusted/world1/monthly/raw/'
dir_pha = 'AUSTRALIA/Adjusted/world1/monthly/FLs.r00/'
dir_stnlist = 'AUSTRALIA/Adjusted/world1/meta/world1_stnlist.tavg'

#dir_raw = 'AUSTRALIA/Unadjusted/world1/monthly/raw/'
#dir_pha = 'AUSTRALIA/Unadjusted/world1/monthly/FLs.r00/'
#dir_stnlist = 'AUSTRALIA/Unadjusted/world1/meta/world1_stnlist.tavg'

raw_files = sorted(glob.glob(dir_cwd+dir_raw+'*.raw.tavg'))
pha_files = sorted(glob.glob(dir_cwd+dir_pha+'*.FLs.r00.tavg'))
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
# LOAD: raw station absolute temperatures (Tx100) in GHCNm-v3 format --> df_raw
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

    # Check for empty station file

    if len(dates) == 0: print(stationcodes[k])

    # Create station dataframe

    df = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = dates
    df['stationcode'] = len(dates)*[stationcodes[k]]

    # Store monthly values
       
    for j in range(12): df[df.columns[j+2]] = [ obs[i][j] for i in range(len(obs)) ]

    # Replace monthly fill value -9999 with np.nan    

    for j in range(12): df[df.columns[j+2]].replace(-9999, np.nan, inplace=True)

    # Apply /100 scale factor    

    for j in range(12): df[df.columns[j+2]] = df[df.columns[j+2]]/100.0

    # Calculate station normals
    
    normals = df[(df['year']>1960) & (df['year']<1991)].mean()[2:]
    if flag_absolute == True: normals = np.zeros(12)
    
    # Calculate station aanomaly timeseries as required

    for j in range(12): df[df.columns[j+2]] = df[df.columns[j+2]] - normals[j]

    # Append station month array to dataframe

    df_raw = df_raw.copy().append([df],ignore_index=True)

# EXTRACT: timeseries, fill in missing years and store in dataframe --> dg_raw

t = pd.date_range(start=str(df_raw.min().year), end=str(df_raw.max().year+1), freq='M')
dg_raw = pd.DataFrame( columns=df_raw['stationcode'].unique(), index=t)       
for i in range(n):
    da = df_raw[df_raw['stationcode']==df_raw['stationcode'].unique()[i]].iloc[:,1:]
    ts_monthly = []
    k = 0
    for j in range(da['year'].iloc[0], da['year'].iloc[-1]+1):
        if da['year'].iloc[k] == j:
            ts_monthly += list(da.iloc[k,1:])
            k += 1
        else:
            ts_monthly += list(12*[np.nan])    
#   ts_monthly = np.array(da.groupby('year').mean().iloc[:,0:]).ravel() 
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')       
    db = pd.DataFrame(ts_monthly,index=t_monthly)
    dg_raw[dg_raw.columns[i]] = db

#    if (len(da)-(da.year.iloc[-1]-da.year.iloc[0]+1) > 0) | (len(da)-(da.year.iloc[-1]-da.year.iloc[0]+1) < 0):
#        print(i, len(da)-(da.year.iloc[-1]-da.year.iloc[0]+1))

dg_raw['mean'] = dg_raw.mean(axis=1)

#------------------------------------------------------------------------------
# LOAD: PHA station adjusted absolute temperatures (Tx100) in GHCNm-v3 format --> df_pha
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

    # Check for empty station file

    if len(dates) == 0: print(stationcodes[k])
#   if len(dates) == 0: continue

    # Create station dataframe: NB: stations with no data drop out

    df = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = dates
    df['stationcode'] = len(dates)*[stationcodes[k]]
    
    # Store monthly values
    
    for j in range(12): df[df.columns[j+2]] = [ obs[i][j] for i in range(len(obs)) ]

    # Replace monthly fill value -9999 with np.nan    

    for j in range(12): df[df.columns[j+2]].replace(-9999, np.nan, inplace=True)

    # Apply /100 scale factor    

    for j in range(12): df[df.columns[j+2]] = df[df.columns[j+2]]/100.0

    # Calculate station normals

    normals = df[(df['year']>1960) & (df['year']<1991)].mean()[2:]
    if flag_absolute == True:
       normals = np.zeros(12)
    
    # Calculate station aanomaly timeseries as required

    for j in range(12): df[df.columns[j+2]] = df[df.columns[j+2]] - normals[j]

    # Append station month array to dataframe

    df_pha = df_pha.copy().append([df],ignore_index=True)
    
# EXTRACT: timeseries, fill in missing years and store in dataframe --> dg_raw

t = pd.date_range(start=str(df_pha.min().year), end=str(df_pha.max().year+1), freq='M')
dg_pha = pd.DataFrame( columns=df_pha['stationcode'].unique(), index=t)   
for i in range(n):
    da = df_pha[df_pha['stationcode']==df_pha['stationcode'].unique()[i]].iloc[:,1:]
    ts_monthly = []
    k = 0
    for j in range(da['year'].iloc[0], da['year'].iloc[-1]+1):
        if da['year'].iloc[k] == j:
            ts_monthly += list(da.iloc[k,1:])
            k += 1
        else:
            ts_monthly += list(12*[np.nan])
#   ts_monthly = np.array(da.groupby('year').mean().iloc[:,0:]).ravel() 
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')       
    db = pd.DataFrame(ts_monthly,index=t_monthly)
    dg_pha[dg_pha.columns[i]] = db

#    if (len(da)-(da.year.iloc[-1]-da.year.iloc[0]+1) > 0) | (len(da)-(da.year.iloc[-1]-da.year.iloc[0]+1) < 0):        
#        print(i, len(da)-(da.year.iloc[-1]-da.year.iloc[0]+1))

dg_pha['mean'] = dg_pha.mean(axis=1)

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# PLOT: raw (mean) v adjusted (mean)
#------------------------------------------------------------------------------

print('plotting mean raw versus PHA-adjusted ...')

mean_raw = dg_raw['mean'].rolling(12, center=True).mean()
mean_pha = dg_pha['mean'].rolling(12, center=True).mean()
#mean_raw = dg_raw['mean']
#mean_pha = dg_pha['mean']
diff_raw_pha = mean_raw - mean_pha
mask = np.isfinite(diff_raw_pha) 

figstr = 'raw-v-pha-means.png'
titlestr = 'Australia (unadjusted): mean raw versus PHA (1300 stations)'
                 
fig, ax = plt.subplots(figsize=(15,10))          
plt.plot(mean_raw.index, mean_raw, marker='.', markersize=10, color='blue', alpha=0.5, ls='-', lw=1.0, zorder=1, label='raw: 12m MA')
plt.plot(mean_pha.index, mean_pha, marker='.', markersize=10, color='red', alpha=0.5, ls='-', lw=1.0, zorder=2, label='PHA: 12m MA')
#plt.vlines(mean_raw.index, mean_raw, mean_pha, colors='lightgrey', zorder=0)
#plt.vlines(mean_raw.index[mask], mean_raw[mask], mean_pha[mask], colors='lightgrey', zorder=0)
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', markerscale=2, fontsize=fontsize)
plt.xlabel('Year', fontsize=fontsize)
if flag_absolute == True:
    plt.ylabel(r'Absolute temperature [°C]', fontsize=fontsize)
else:
    plt.ylabel(r'Temperature anomaly (from 1961-1990) [°C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close(fig)

#------------------------------------------------------------------------------
# PLOT: raw and PHA-adjusted temporal coverage 
#------------------------------------------------------------------------------

print('plotting raw and PHA-adjusted temporal coverage ...')

raw_min = df_raw.groupby('stationcode').min()['year']
raw_max = df_raw.groupby('stationcode').max()['year']
pha_min = df_pha.groupby('stationcode').min()['year']
pha_max = df_pha.groupby('stationcode').max()['year']

figstr = 'raw-v-pha-temporal-coverage.png'
titlestr = 'Australia (unadjusted): temporal coverage (1300 stations)'
                 
fig, ax = plt.subplots(figsize=(15,10))          
plt.step(np.arange(len(raw_min)), raw_min, color='blue', alpha=1.0, label='raw')
plt.step(np.arange(len(raw_max)), raw_max, color='blue', alpha=0.5)
plt.vlines(np.arange(len(raw_min)), raw_min, raw_max, color='blue', alpha=0.1)
plt.step(np.arange(len(pha_min)), pha_min, color='red', alpha=1.0, label='PHA-adjusted')
plt.step(np.arange(len(pha_max)), pha_max, color='red', alpha=0.5)
plt.vlines(np.arange(len(pha_min)), pha_min, pha_max, color='red', alpha=0.1)
ax.set_ylim(1825,2025)
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', fontsize=fontsize)
plt.ylabel(r'Year', fontsize=fontsize)
plt.xlabel(r'Station rank by coverage', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close(fig)

#------------------------------------------------------------------------------
# PLOT: raw (mean) - adjusted (mean)
#------------------------------------------------------------------------------

print('plotting difference of mean raw minus mean PHA-adjusted ...')

figstr = 'raw-v-pha-means-diff.png'
titlestr = 'Australia (unadjusted): mean raw-PHA adjustments (1300 stations)'
                 
fig, ax = plt.subplots(figsize=(15,10))          
plt.plot(mean_raw.index[mask], diff_raw_pha[mask], '.', markersize=10, color='black', alpha=0.5, ls='-', lw=1.0)
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
plt.tick_params(labelsize=fontsize)    
plt.xlabel('Year', fontsize=fontsize)
if flag_absolute == True:
    plt.ylabel(r'Absolute temperature adjustment [°C]', fontsize=fontsize)
else:
    plt.ylabel(r'Temperature anomaly adjumstment (from 1961-1990) [°C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close(fig)

#------------------------------------------------------------------------------
# PLOT: histogram of adjustments
#------------------------------------------------------------------------------

print('plotting (missing middle) histogram of raw minus PHA-adjusted differences ...')
    
adj_all = []
for i in range(n):
#    raw_ts_yearly = dg_raw.iloc[:,i].rolling(12).mean()
#    pha_ts_yearly = dg_pha.iloc[:,i].rolling(12).mean()
    raw_ts_yearly = dg_raw.iloc[:,i]
    pha_ts_yearly = dg_pha.iloc[:,i]
    adj_ts_yearly = raw_ts_yearly-pha_ts_yearly
    adj_all.append(adj_ts_yearly)    
adj_all = np.array(adj_all).flatten()
mask = (np.isfinite(adj_all)) & (np.abs(adj_all)>0.0)
adjustments = adj_all[mask]

xmin = -6.0; xmax = 6.0
#bins = int((xmax-xmin)*10+1)
bins = 301
x = np.linspace(xmin,xmax,301)

figstr = 'adjustment-histogram.png'
titlestr = 'Australia (unadjusted): histogram of raw-PHA adjustments (1300 stations)'
                 
fig, ax = plt.subplots(figsize=(15,10))     
kde = stats.gaussian_kde(adjustments, bw_method='silverman'); 
plt.hist(adjustments, density=False, bins=bins, alpha=1.0, color='lightgrey', label='bin counts')
ax1 = plt.gca()
ymin,ymax = ax1.get_ylim()
#plt.plot(x, kde(x)*ymax, color='red', lw=3, label='Kernel density estimate (KDE)')
#ax2 = ax.twinx()
#ax2.plot(x, kde(x), color='red', lw=3)
if flag_absolute == True:
    ax1.set_xlabel(r'Absolute temperature adjustment [°C]', fontsize=fontsize)
else:
    ax1.set_xlabel(r'Temperature anomaly (from 1961-1990) adjustment [°C]', fontsize=fontsize)
ax1.set_ylabel('Count', fontsize=fontsize)
ax1.set_xlim(xmin,xmax)
ax1.set_ylim(ymin,ymax)
ax1.xaxis.grid(True, which='major')      
ax1.tick_params(labelsize=fontsize)    
#ax2.set_ylabel('Kernel density estimate (KDE)', fontsize=fontsize, color='red')
#ax2.tick_params(labelsize=16, colors='red')    
#ax2.spines['right'].set_color('red')
plt.legend(loc='upper right', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close(fig)

#------------------------------------------------------------------------------
# PLOT: raw versus PHA-adjusted per station
#------------------------------------------------------------------------------

print('plotting random seletion of station raw versus PHA-adjusted ...')

nstations = 10
draws = random.sample(range(n), nstations)
for i in draws:

    raw_ts_yearly = dg_raw.iloc[:,i].rolling(12, center=True).mean()
    pha_ts_yearly = dg_pha.iloc[:,i].rolling(12, center=True).mean()
    raw_ts = dg_raw.iloc[:,i]
    pha_ts = dg_pha.iloc[:,i]
    mask = np.isfinite(raw_ts_yearly - pha_ts_yearly) 
    
    ds = pd.DataFrame({'raw':raw_ts_yearly, 'pha':pha_ts_yearly})
    
    figstr = stationcodes[i] + '-raw-v-pha.png'
    titlestr = stationcodes[i] + ': raw versus PHA'
                 
    fig, ax = plt.subplots(figsize=(15,10))          
    plt.plot(raw_ts.index, raw_ts, marker='.', markersize=10, color='blue', alpha=0.1, ls='-', lw=1.0)
    plt.plot(pha_ts.index, pha_ts, marker='.', markersize=10, color='red', alpha=0.1, ls='-', lw=1.0)
    plt.plot(raw_ts_yearly.index, raw_ts_yearly, marker='.', markersize=10, color='blue', alpha=0.5, ls='-', lw=1.0, label='raw: 12m MA')    
    plt.plot(pha_ts_yearly.index, pha_ts_yearly, marker='.', markersize=10, color='red', alpha=0.5, ls='-', lw=1.0, label='PHA: 12m MA')        
#    plt.vlines(raw_ts_yearly.index[mask], raw_ts_yearly[mask], pha_ts_yearly[mask], colors='lightgrey', zorder=0)
    ax.xaxis.grid(True, which='major')      
    ax.yaxis.grid(True, which='major')  
    plt.tick_params(labelsize=16)    
    plt.legend(loc='upper left', fontsize=fontsize)
    plt.xlabel('Year', fontsize=fontsize)
    if flag_absolute == True:
        plt.ylabel(r'Absolute temperature [°C]', fontsize=fontsize)
    else:
        plt.ylabel(r'Temperature anomaly (from 1961-1990) [°C]', fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr, dpi=300)
    plt.close(fig)
    
#------------------------------------------------------------------------------
print('** END')

