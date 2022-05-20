#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: pha-backend.py
#------------------------------------------------------------------------------
# Version 0.6
# 20 May, 2022
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

# Stats libraries:
import scipy
import scipy.stats as stats    
from sklearn.preprocessing import StandardScaler

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import seaborn as sns; sns.set()
#import matplotlib.cm as cm
#import cmocean
#from matplotlib import colors as mcol
#from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
#import matplotlib.dates as mdates
#import matplotlib.colors as mcolors
#import matplotlib.ticker as mticker

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
flag_absolute = True
flag_comparison_draws = True

dir_cwd = '/local/cqz20mbu/Documents/REPOS/pha-backend/'

#region = 'ICELAND'
#region = 'AUSTRALIA'
region = 'NORWAY'

dir_raw = region + '/Adjusted/world1/monthly/raw/'
dir_pha = region + '/Adjusted/world1/monthly/FLs.r00/'
#dir_pha = region + '/Adjusted/world1/monthly/WMs.r00/'
dir_stnlist = region + '/Adjusted/world1/meta/world1_stnlist.tavg'

raw_files = sorted(glob.glob(dir_cwd+dir_raw+'*.raw.tavg'))
pha_files = sorted(glob.glob(dir_cwd+dir_pha+'*.FLs.r00.tavg'))
#pha_files = sorted(glob.glob(dir_cwd+dir_pha+'*.WMs.r00.tavg'))
stnlist = glob.glob(dir_cwd+dir_stnlist)[0]

#------------------------------------------------------------------------------
# LOAD: stnlist
#------------------------------------------------------------------------------

f = open(stnlist)
lines = f.readlines()
stationcodes = []
for i in range( len(lines) ):
    words = lines[i].split()    
    stationcode = words[0]
    stationcodes.append(stationcode)        
f.close()    

#------------------------------------------------------------------------------
# LOAD: raw station absolute temperatures (Tx100) in GHCNm-v3 format --> df_raw
#------------------------------------------------------------------------------

print('loading raw files ...')

df_raw = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
for k in range( len(raw_files) ):

    f = open(raw_files[k])
    lines = f.readlines()
    dates = []
    obs = []
    for i in range( len(lines) ):
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

    stationcodes_raw_files = [ raw_files[i][-15:-9] for i in range(len(raw_files)) ]

    # Check for empty station file

    if len(dates) == 0: print('EMPTY:', stationcodes_raw_files[k])

    # Create station dataframe

    df = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = dates
    df['stationcode'] = len(dates)*[stationcodes_raw_files[k]]

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

t = pd.date_range(start=str(df_raw.min().year), end=str(df_raw.max().year+1), freq='MS')[0:-1]
dg_raw = pd.DataFrame( columns=df_raw['stationcode'].unique(), index=t)       
for i in range( len(df_raw.stationcode.unique()) ):
    da = df_raw[df_raw['stationcode']==df_raw['stationcode'].unique()[i]].iloc[:,1:]
    ts_monthly = []
    k = 0
    for j in range(da['year'].iloc[0], da['year'].iloc[-1]+1):
        if da['year'].iloc[k] == j:
            ts_monthly += list(da.iloc[k,1:])
            k += 1
        else:
            ts_monthly += list(12*[np.nan])    
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='MS')       
    db = pd.DataFrame(ts_monthly,index=t_monthly)
    dg_raw[dg_raw.columns[i]] = db

dg_raw['mean'] = np.nanmean( dg_raw, axis=1)
n_raw = df_raw.stationcode.unique().shape[0]

#------------------------------------------------------------------------------
# LOAD: PHA station adjusted absolute temperatures (Tx100) in GHCNm-v3 format --> df_pha
#------------------------------------------------------------------------------

print('loading PHA files ...')

df_pha = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
for k in range( len(pha_files) ): 
    
    f = open(pha_files[k])
    lines = f.readlines()
    dates = []
    obs = []
    for i in range( len(lines) ):
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

    stationcodes_pha_files = [ pha_files[i][-19:-13] for i in range(len(pha_files)) ]

    # Check for empty station file

    if len(dates) == 0: print('EMPTY:', stationcodes_pha_files[k])

    # Create station dataframe: NB: stations with no data drop out

    df = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = dates
    df['stationcode'] = len(dates)*[stationcodes_pha_files[k]]
    
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

t = pd.date_range(start=str(df_pha.min().year), end=str(df_pha.max().year+1), freq='MS')[0:-1]
dg_pha = pd.DataFrame( columns=df_pha['stationcode'].unique(), index=t)   
for i in range( len(df_pha.stationcode.unique()) ):
    da = df_pha[df_pha['stationcode']==df_pha['stationcode'].unique()[i]].iloc[:,1:]
    ts_monthly = []
    k = 0
    for j in range(da['year'].iloc[0], da['year'].iloc[-1]+1):
        if da['year'].iloc[k] == j:
            ts_monthly += list(da.iloc[k,1:])
            k += 1
        else:
            ts_monthly += list(12*[np.nan])
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='MS')       
    db = pd.DataFrame(ts_monthly,index=t_monthly)            
    dg_pha[dg_pha.columns[i]] = db

dg_pha['mean'] = dg_pha.mean(axis=1)
n_pha = df_pha.stationcode.unique().shape[0]

#==============================================================================
# STATS
#==============================================================================

mean_raw = dg_raw['mean'].rolling(12, center=True).mean() 
mean_pha = dg_pha['mean'].rolling(12, center=True).mean() 
mean_diff = np.subtract( dg_raw['mean'], dg_pha['mean'] )

min_raw = df_raw.groupby('stationcode').min()['year']
max_raw = df_raw.groupby('stationcode').max()['year']
min_pha = df_pha.groupby('stationcode').min()['year']
max_pha = df_pha.groupby('stationcode').max()['year']
da = pd.DataFrame( {'min_raw':min_raw, 'max_raw':max_raw}, index=min_raw.index ) 
db = pd.DataFrame( {'min_pha':min_pha, 'max_pha':max_pha}, index=min_pha.index ) 
dc = da.merge( db, how='left', on='stationcode')

#==============================================================================
# PLOTS
#==============================================================================

#------------------------------------------------------------------------------
# PLOT: raw (mean) v PHA (mean) + station running averages
#------------------------------------------------------------------------------

print('plotting mean raw versus PHA ...')

figstr = 'raw-v-pha-means.png'
titlestr = region.title() + ': raw versus PHA (' + str( n_pha ) + ' stations)'
                 
fig, ax = plt.subplots(figsize=(15,10))          
plt.plot(dg_raw.index, dg_raw.rolling(12).mean(), '.', alpha=0.05) 
#plt.plot(dg_pha.index, dg_pha.rolling(12).mean(), '.', alpha=0.05)
plt.plot(dg_raw.index, dg_raw['mean'].rolling(24).mean(), marker='o', markersize=1, color='blue', alpha=0.5, ls='-', lw=3, label='raw mean (12m MA)' )
plt.plot(dg_pha.index, dg_pha['mean'].rolling(24).mean(), marker='o', markersize=1, color='red', alpha=0.5, ls='-', lw=3, label='PHA mean (12m MA)' )
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
# PLOT: raw (mean) - PHA (mean)
#------------------------------------------------------------------------------

print('plotting difference of mean raw - mean PHA ...')

figstr = 'raw-v-pha-means-diff.png'
titlestr = region.title() + ': raw-PHA mean (' + str(n_pha) + ' stations)'
                 
fig, ax = plt.subplots(figsize=(15,10))          
plt.plot(mean_diff.index, mean_diff, '.', markersize=10, color='black', alpha=0.5, ls='-', lw=1.0)
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
# PLOT: raw and PHA temporal coverage 
#------------------------------------------------------------------------------

print('plotting raw and PHA temporal coverage ...')

figstr = 'raw-v-pha-temporal-coverage.png'
titlestr = region.title() + ': temporal coverage (' + str(n_pha) + ' stations)'
                 
fig, ax = plt.subplots(figsize=(15,10))          
plt.fill_between(np.arange(len(dc)), dc.min_raw, dc.max_raw, step='pre', color='blue', alpha=0.2, label='raw')
plt.fill_between(np.arange(len(dc)), dc.min_pha, dc.max_pha, step='pre', color='red', alpha=0.2, label='PHA')
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
# PLOT: histogram of adjustments
#------------------------------------------------------------------------------

print('plotting ("missing middle") histogram of raw minus PHA differences ...')
    
adj_all = []
for i in range( n_pha ):
    
    stationcode = dg_pha.columns[i]  
    adj_ts_yearly = np.subtract( dg_raw[stationcode], dg_pha[stationcode] )
    adj_all.append(adj_ts_yearly)    

adj_all = np.array(adj_all).flatten()
mask = (np.isfinite(adj_all)) & (np.abs(adj_all)>0.0)
#mask = np.isfinite(adj_all)
adjustments = adj_all[mask]

xmin = -2.0; xmax = 2.0
#bins = int((xmax-xmin)*10+1)
bins = 201
x = np.linspace( xmin, xmax, bins)

figstr = 'adjustment-histogram.png'
titlestr = region.title() + ': histogram of raw-PHA adjustments (' + str( n_pha ) + ' stations)'
                 
fig, ax = plt.subplots(figsize=(15,10))     
kde = stats.gaussian_kde( adjustments, bw_method='silverman' ) 
h = plt.hist( adjustments, density=False, bins=bins, alpha=1.0, color='lightgrey', label='bin counts' )
#ax1 = plt.gca(); ymin,ymax = ax1.get_ylim()
ymax = h[0].max()
plt.plot(x, kde(x)*ymax, color='red', lw=3, label='Kernel density estimate (KDE)')
if flag_absolute == True:
    ax.set_xlabel(r'Absolute temperature adjustment [°C]', fontsize=fontsize)
else:
    ax.set_xlabel(r'Temperature anomaly (from 1961-1990) adjustment [°C]', fontsize=fontsize)
ax.set_ylabel('Count', fontsize=fontsize)
ax.set_xlim(xmin,xmax)
#ax1.set_ylim(ymin,ymax)
ax.xaxis.grid(True, which='major')      
ax.tick_params(labelsize=fontsize, colors='k')
#ax.spines['left'].set_color('k')
plt.legend(loc='upper left', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close(fig)

if flag_comparison_draws == True:
    
    #------------------------------------------------------------------------------
    # PLOT: raw versus PHA-adjusted per station
    #------------------------------------------------------------------------------
    
    print('plotting random seletion of station raw versus PHA ...')
    
    ndraws = 10
    draws = random.sample( range( n_pha ), ndraws)
    for i in draws:
    
        stationcode = dg_pha.columns[i]          
        raw_ts_yearly = dg_raw[stationcode].rolling(12, center=True).mean()
        pha_ts_yearly = dg_pha[stationcode].rolling(12, center=True).mean()
        raw_ts = dg_raw[stationcode]
        pha_ts = dg_pha[stationcode]
        
        figstr = stationcode + '-raw-v-pha.png'
        titlestr = stationcode + ': raw versus PHA'
                     
        fig, ax = plt.subplots(figsize=(15,10))          
        plt.plot(raw_ts.index, raw_ts, marker='.', markersize=10, color='blue', alpha=0.1, ls='-', lw=1.0)
        plt.plot(pha_ts.index, pha_ts, marker='.', markersize=10, color='red', alpha=0.1, ls='-', lw=1.0)
        plt.plot(raw_ts_yearly.index, raw_ts_yearly, marker='.', markersize=10, color='blue', alpha=0.5, ls='-', lw=1.0, label='raw: 12m MA')    
        plt.plot(pha_ts_yearly.index, pha_ts_yearly, marker='.', markersize=10, color='red', alpha=0.5, ls='-', lw=1.0, label='PHA: 12m MA')        
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

