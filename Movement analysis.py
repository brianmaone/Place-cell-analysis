# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:11:34 2022

@author: SunLab
"""
#%% Importing packages
import pandas as pd
from pathlib import Path  # to work with dir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics as st
from scipy.ndimage import gaussian_filter
from scipy import signal
import json

#%%
'''
Import saved calcium data before plotting
'''
#%% Indicate day for plotting
'''
MANUAL INPUT REQUIRED
Indicate the session to be plotted, run before plotting
'''
plotday = '0630'
plotevent = CalciumData[plotday]['Events']
indst = CalciumData[plotday]['indst']
indet = CalciumData[plotday]['indet']
sessions = [0,1,2,3,4]
#%% Indicate session
session = 0
plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
'''
MANUAL INPUT REQUIRED
Check labeled frames to get x boundary and grid length
'''
#xbound = 60
#xgridlen = 490/30
#%% Plot movement trajectory
for session in sessions:
    plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
    indmid = int(len(plotsession.index)*0.5)
    plt.figure()
    sns.lineplot(x=plotsession['x'],y=plotsession['y'],sort=False, lw=1).set(title=plotday + ' Trajectory session' + str(session))
    plt.ylim(480,0)
    plt.xlim(0,640)
    plt.savefig(plotday + 'Movement trajectory session' + str(session) + '.png')
    
    plt.figure()
    sns.lineplot(x=plotsession[:indmid]['x'],y=plotsession[:indmid]['y'],sort=False, lw=1).set(title=plotday + ' Trajectory session' + str(session) + ' 1st')
    plt.ylim(480,0)
    plt.xlim(0,640)
    plt.savefig(plotday + 'Movement trajectory session' + str(session) + ' 1st.png')
    
    plt.figure()
    sns.lineplot(x=plotsession[indmid:]['x'],y=plotsession[indmid:]['y'],sort=False, lw=1).set(title=plotday + ' Trajectory session' + str(session) + ' 2nd')
    plt.ylim(480,0)
    plt.xlim(0,640)
    plt.savefig(plotday + 'Movement trajectory session' + str(session) + ' 2nd.png')

#%% Plot speed versus time
for session in sessions:
    plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
    plt.figure()
    sns.lineplot(x=plotsession['Time (s)'],y=plotsession['Speed']).set(title=plotday + ' Speed session' + str(session))
    plt.savefig(plotday + 'Speed session' + str(session) + '.png')

#%% Plot distribution of speed
for session in sessions:
    plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
    plt.figure()
    sns.displot(data=plotsession, x='Speed', kde=True, stat='probability',binwidth=0.5,height=5, aspect=2).set(title=plotday + ' Speed distribution session' + str(session))
    plt.savefig(plotday + 'Speed distribution session' + str(session) + '.png', bbox_inches='tight')

#%% Plot receptive fields
for session in sessions:
    plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
    for cell in plotevent.columns[1:]:
        plt.figure()
        ax = plt.subplot(111)
        ax.axis('off')
        indeventspre = plotevent[indst[session]:indet[session]+1].loc[plotevent[cell]> 0.0].index.tolist()
        indevents = [i - indst[session] for i in indeventspre]
        sns.lineplot(x=plotsession['x'],y=plotsession['y'],sort=False, lw=1).set(title= plotday + ' Events session' + str(session) + cell)
        plt.plot(plotsession.loc[indevents,'x'].loc[plotsession['Speed']>0.5],plotsession.loc[indevents,'y'].loc[plotsession['Speed']>0.5],'ro',markersize=5)
        plt.ylim(480,0)
        plt.xlim(0,640)
        plt.savefig(plotday + cell + ' Events session' + str(session) + '.png', bbox_inches='tight')
#%% Plot speed with calcium traces
'''
'''
fig=plt.figure()
ax = plt.subplot(111)
for n,i in enumerate(sessions['dfs0'].columns[[13,39,42,43,56,61,65,76,88,115,126]]):
    ax.plot(sessions['dfs0']['Time (s)'],sessions['dfs0'][i]*2+n*2,label=i)
ax.plot(sessions['dfs0']['Time (s)'],sessions['dfs0']['Speed']*2/np.max(sessions['dfs0']['Speed'])+n*2+5,'k',label='Speed')
plt.xlabel('Time(s)')
plt.title('Session3')
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels),loc='upper right', bbox_to_anchor=(1.25, 1.05), fancybox=True, shadow=True)
ax.get_yaxis().set_visible(False)
#plt.savefig('Session3 Traces' + '.png')
#%% Plot traces with events
for session in sessions:
    plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
    fig=plt.figure()
    ax = plt.subplot(111)
    plt.xlabel('Time(s)')
    plt.title(plotday + ' session'+str(session))
    ax.get_yaxis().set_visible(False)
    for n,cell in enumerate([' C56', ' C60', ' C63',' C76']):
        plt.plot(plotsession['Time (s)'], plotsession[cell]+n*1.5)
        indeventspre = plotevent[indst[session]:indet[session]+1].loc[plotevent[cell]> 0.0].index.tolist()
        indevents = [i - indst[session] for i in indeventspre]
        plt.plot(plotsession.loc[indevents],[1.2+n*1.5]*len(indevents),'bo',markersize=1)
    plt.savefig(plotday + ' session'+str(session) + '.png', bbox_inches='tight')
