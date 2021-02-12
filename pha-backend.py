#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: pha-backend.py
#------------------------------------------------------------------------------
# Version 0.1
# 11 February, 2021
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
import os
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

#------------------------------------------------------------------------------
# LOAD: Iceland station absolute temperatures (Tx10)
#------------------------------------------------------------------------------

nheader = 0
stationcode = 'PW100482715'
f = open(stationcode+'.raw.tavg')
#f = open(stationcode+'.FLs.r00.tavg')
#f = open(stationcode+'.WMs.r00.tavg')
lines = f.readlines()
dates = []
obs = []
for i in range(nheader,len(lines)):
    words = lines[i].split()    
    date = int(words[1])
    val = (len(words)-2)*[None]
    for j in range(len(val)):                            
        try: val[j] = int(words[j+2])
        except:                    
            pass
    dates.append(date)        
    obs.append(val) 
f.close()    
dates = np.array(dates)
obs = np.array(obs)

df = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
# df['mean'] = da[da.columns[range(1,13)]].mean(axis=1)

df['stationcode'] = len(dates)*[stationcode]
df['year'] = dates
for j in range(12):        
    df[df.columns[j+2]] = [ obs[i][j] for i in range(len(obs)) ]

# Replace monthly fill value -999 with np.nan    
for j in range(12):        
    df[df.columns[j+2]].replace(-9999, np.nan, inplace=True)
    
# Apply /100 scale factor    
for j in range(12):       
    df[df.columns[j+2]] = df[df.columns[j+2]]/100.0

df_ts_monthly = np.array(df.groupby('year').mean().iloc[:,1:]).ravel() 
df_t_monthly = pd.date_range(start=str(df['year'].iloc[0]), periods=len(df_ts_monthly), freq='M')   

# --------------


nheader = 0
stationcode = 'PW100482715'
f = open(stationcode+'.FLs.r00.tavg')
lines = f.readlines()
dates = []
obs = []
for i in range(nheader,len(lines)):
    words = lines[i].split()    
    date = int(words[0][-4:])
    val = (len(words)-1)*[None]
    for j in range(len(val)):                            
        try: val[j] = int(words[j+1].rstrip('E'))
        except:                    
            pass
    dates.append(date)        
    obs.append(val) 
f.close()    
dates = np.array(dates)
obs = np.array(obs)

df_FL = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
# df['mean'] = da[da.columns[range(1,13)]].mean(axis=1)

df_FL['stationcode'] = len(dates)*[stationcode]
df_FL['year'] = dates
for j in range(12):        
    df_FL[df_FL.columns[j+2]] = [ obs[i][j] for i in range(len(obs)) ]

# Replace monthly fill value -999 with np.nan    
for j in range(12):        
    df_FL[df_FL.columns[j+2]].replace(-9999, np.nan, inplace=True)
    
# Apply /100 scale factor    
for j in range(12):       
    df_FL[df_FL.columns[j+2]] = df_FL[df_FL.columns[j+2]]/100.0

df_FL_ts_monthly = np.array(df_FL.groupby('year').mean().iloc[:,1:]).ravel() 
df_FL_t_monthly = pd.date_range(start=str(df_FL['year'].iloc[0]), periods=len(df_FL_ts_monthly), freq='M')   

# --------------


nheader = 0
stationcode = 'PW100482715'
f = open(stationcode+'.WMs.r00.tavg')
lines = f.readlines()
dates = []
obs = []
for i in range(nheader,len(lines)):
    words = lines[i].split()    
    date = int(words[0][-4:])
    val = (len(words)-1)*[None]
    for j in range(len(val)):                            
        try: val[j] = int(words[j+1].rstrip('E'))
        except:                    
            pass
    dates.append(date)        
    obs.append(val) 
f.close()    
dates = np.array(dates)
obs = np.array(obs)

df_WM = pd.DataFrame(columns=['stationcode','year','1','2','3','4','5','6','7','8','9','10','11','12'])
# df['mean'] = da[da.columns[range(1,13)]].mean(axis=1)

df_WM['stationcode'] = len(dates)*[stationcode]
df_WM['year'] = dates
for j in range(12):        
    df_WM[df_WM.columns[j+2]] = [ obs[i][j] for i in range(len(obs)) ]

# Replace monthly fill value -999 with np.nan    
for j in range(12):        
    df_WM[df_WM.columns[j+2]].replace(-9999, np.nan, inplace=True)
    
# Apply /100 scale factor    
for j in range(12):       
    df_WM[df_WM.columns[j+2]] = df_WM[df_WM.columns[j+2]]/100.0

df_WM_ts_monthly = np.array(df_WM.groupby('year').mean().iloc[:,1:]).ravel() 
df_WM_t_monthly = pd.date_range(start=str(df_WM['year'].iloc[0]), periods=len(df_WM_ts_monthly), freq='M')   



diff_FL = df_ts_monthly - df_FL_ts_monthly
diff_WM = df_ts_monthly - df_WM_ts_monthly


#------------------------------------------------------------------------------
# LOAD: normals5
#------------------------------------------------------------------------------

# normals = pd.read_pickle('df_normals.pkl', compression='bz2')

nheader = 0
f = open('normals5.GloSAT.prelim03_FRYuse_ocPLAUS1_iqr3.600reg0.3_19411990_MIN15_OCany_19611990_MIN15_PERDEC00_NManySDreq.txt')
lines = f.readlines()
stationcodes = []
sourcecodes = []
obs = []
for i in range(nheader,len(lines)):
    words = lines[i].split()    
    stationcode = words[0][0:6]
    sourcecode = int(words[17])
    val = (12)*[None]
    for j in range(12):                         
        try: val[j] = float(words[j+5])
        except:                                    
            pass
    stationcodes.append(stationcode)
    sourcecodes.append(sourcecode)
    obs.append(val)     
f.close()    
obs = np.array(obs)

dn = pd.DataFrame(columns=['stationcode','sourcecode','1','2','3','4','5','6','7','8','9','10','11','12'])

dn['stationcode'] = stationcodes
dn['sourcecode'] = sourcecodes
for j in range(12):        
    dn[dn.columns[j+2]] = [ obs[i][j] for i in range(len(obs)) ]

# Replace monthly fill value -999 with np.nan    
for j in range(12):        
    dn[dn.columns[j+2]].replace(-999., np.nan, inplace=True)

# Filter out stations with missing normals

df_normals = df[df['stationcode'].isin(dn[dn['sourcecode']>1]['stationcode'])].reset_index()
dn_normals = dn[dn['stationcode'].isin(df_normals['stationcode'])].reset_index()

df_anom = df_normals.copy()
for i in range(len(df_normals)):
    normals = dn_normals[dn_normals['stationcode']==df_normals['stationcode'][i]]
    for j in range(1,13):            
        df_anom[str(j)][i] = df_normals[str(j)][i] - normals[str(j)]
df = df_anom.copy()

# Monthly mean

t = pd.date_range(start=str(df['year'].min()), end=str(df['year'].max()+1), freq='M')   
dg = pd.DataFrame(columns=df['stationcode'].unique(),index=t)   

n = len(df['stationcode'].unique())
for i in range(n):
    da = df[df['stationcode']==df['stationcode'].unique()[i]]
    ts_monthly = np.array(da.groupby('year').mean().iloc[:,1:]).ravel() 
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')   
    db = pd.DataFrame({dg.columns[i]:ts_monthly},index=t_monthly)
    dg[dg.columns[i]] = db

dg['mean'] = dg.mean(axis=1)

#------------------------------------------------------------------------------
# PLOT: ALL STATION TIMESERIES
#------------------------------------------------------------------------------

figstr = 'iceland-timeseries-anomalies-yearly-mean.png'
titlestr = 'Iceland21.postmerge (normals)'
             
n = len(df['stationcode'].unique())
colors = cmocean.cm.thermal(np.linspace(0.05,0.95,n)) 
hexcolors = [ "#{:02x}{:02x}{:02x}".format(int(colors[i][0]*255),int(colors[i][1]*255),int(colors[i][2]*255)) for i in range(len(colors)) ]

fig, ax = plt.subplots(figsize=(15,10))          

for i in range(n):
    da = df[df['stationcode']==df['stationcode'].unique()[i]]
    ts_monthly = np.array(da.groupby('year').mean().iloc[:,1:]).ravel() 
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')           
#   plt.scatter(t_monthly, ts_monthly, marker='s', lw=0.2, color=hexcolors[i], alpha=0.8, label=df['stationcode'].unique()[i])
    plt.plot(t_monthly, ts_monthly, lw=0.8, color=hexcolors[i], alpha=0.8, label=df['stationcode'].unique()[i])
plt.plot(dg['mean'].dropna().rolling(12).mean(), marker='.', ls='None', color='black', alpha=1.0, label='Annual mean')
ax.xaxis.grid(True, which='major')      
ax.yaxis.grid(True, which='major')  
plt.tick_params(labelsize=16)    
plt.legend(loc='upper left', bbox_to_anchor=(1.04,1), ncol=3, fontsize=8)
plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'Temperature anomaly (from 1961-1990) [K]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.subplots_adjust(right=0.7)
plt.savefig(figstr)
plt.close(fig)

#------------------------------------------------------------------------------
# CALL RBEAST FOR EACH STATION
#------------------------------------------------------------------------------

for i in range(n):

    da = df[df['stationcode']==df['stationcode'].unique()[i]]
    ts_monthly = np.array(da.groupby('year').mean().iloc[:,1:]).ravel() 
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')           
    dg = pd.DataFrame({'t':t_monthly,'ts':ts_monthly})
    dg = dg.replace(r'^\s+$', np.nan, regex=True)
    dg.to_csv('dg.csv', index=False)
    code = df['stationcode'].unique()[i]

    command = '/usr/bin/Rscript'
#   path2rscript = '~/Desktop/Rbeast/rbeast_frontend.R'
    path2rscript = 'rbeast_frontend.R'
    args = [code]
    cmd = [command, path2rscript] + args
    x = subprocess.call(cmd, universal_newlines=True)
    # x = subprocess.check_output(cmd, universal_newlines=True)
    # subprocess.call ("Rscript --vanilla path2rscript.R")
    out_seasonal = pd.read_csv('out_seasonal.csv')
    out_seasonal_prob = pd.read_csv('out_seasonal_prob.csv')
    out_trend = pd.read_csv('out_trend.csv')
    out_trend_prob = pd.read_csv('out_trend_prob.csv')

    titlestr = 'Rbeast: ' + df['stationcode'].unique()[i]
    figstr = "rbeast-iceland-fit-" + df['stationcode'].unique()[i] + '.png'

    fig, axes = plt.subplots(5,1,sharex=True, figsize=(15,10))      
    axes[0].set_title(titlestr, fontsize=fontsize)
    axes[0].plot(t_monthly, ts_monthly, lw=0.8, color=hexcolors[i], alpha=1.0, label='timeseries')
    axes[0].xaxis.grid(True, which='major')      
    axes[0].yaxis.grid(True, which='major')  
    axes[0].legend(loc='upper left', fontsize=8)
    axes[1].plot(t_monthly, out_trend, lw=0.8, color='blue', alpha=1.0, label='trend')
    axes[1].xaxis.grid(True, which='major')      
    axes[1].yaxis.grid(True, which='major')  
    axes[1].legend(loc='upper left', fontsize=8)
    axes[2].plot(t_monthly, out_trend_prob, lw=0.8, color='grey', alpha=1.0, label='P(trend)')
    axes[2].xaxis.grid(True, which='major')      
    axes[2].yaxis.grid(True, which='major')  
    axes[2].legend(loc='upper left', fontsize=8)
    axes[3].plot(t_monthly, out_seasonal, lw=0.8, color='teal', alpha=1.0, label='seasonal')
    axes[3].xaxis.grid(True, which='major')      
    axes[3].yaxis.grid(True, which='major')  
    axes[3].legend(loc='upper left', fontsize=8)
    axes[4].plot(t_monthly, out_seasonal_prob, lw=0.8, color='grey', alpha=1.0, label='P(seasonal)')
    axes[4].xaxis.grid(True, which='major')      
    axes[4].yaxis.grid(True, which='major')  
    axes[4].legend(loc='upper left', fontsize=8)
    axes[4].set_xlabel('Year', fontsize=fontsize)
#    axes[4].tick_params(labelsize=16)    
    plt.savefig(figstr)
    plt.close(fig)

#------------------------------------------------------------------------------
print('** END')

