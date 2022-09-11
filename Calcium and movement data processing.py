#!/usr/bin/env python
# coding: utf-8

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
# Create data structure to store generated data
'''
Do not run as this will clear all stored values
'''
CalciumData = {}
#%% Read csv files
'''
MANUAL INPUT REQUIRED
Manuever to folder that contains the csv files
Modify file paths to read the corresponding csv files
'''
day = '0805'
CalciumData[day]={}
dfmotion0 = pd.read_csv('Video2022-08-05T10_58_23DLC_resnet50_20220630Jul4shuffle1_500000.csv', low_memory=False)
dfmotion1 = pd.read_csv('Video2022-08-05T11_22_19DLC_resnet50_20220630Jul4shuffle1_500000.csv', low_memory=False)
dfmotion2 = pd.read_csv('Video2022-08-05T11_46_40DLC_resnet50_20220630Jul4shuffle1_500000.csv', low_memory=False)
dfmotion3 = pd.read_csv('Video2022-08-05T12_11_21DLC_resnet50_20220630Jul4shuffle1_500000.csv', low_memory=False)
dfmotion4 = pd.read_csv('Video2022-08-05T12_38_57DLC_resnet50_20220630Jul4shuffle1_500000.csv', low_memory=False)
dfmotions = [dfmotion0,dfmotion1,dfmotion2,dfmotion3,dfmotion4] # Modify to match number of sessions
dfcal = pd.read_csv('0805TracesDN.csv', low_memory=False)
dfgpio = pd.read_csv('0805GPIO.csv', low_memory=False)
dfevent = pd.read_csv('0805Events.csv', low_memory=False)

#%% Data cleaning

# Processing GPIO
dfgpio[['Time (s)',' Value']] = dfgpio[['Time (s)',' Value']].astype(float)

# Processing calcium traces
# Drop rejected cells and unwanted row
for i in dfcal.columns:
    if dfcal[i][0] == ' rejected':
        dfcal.drop(columns=[i],inplace=True)
dfcal.drop(index=0,inplace=True)
dfcal = dfcal.reset_index(drop=True) # Reset index
dfcal = dfcal.astype(float) # Convert values to float
dfcal.rename(columns = {' ':'Time (s)'}, inplace = True)

# Generate normalized raw traces and save
dfcalN = dfcal.copy()
dfcalN.iloc[:,1:]=dfcalN.iloc[:,1:]/(dfcalN.iloc[:,1:].max())
CalciumData[day]['Raw Traces'] = dfcalN.to_json(orient='columns')

# Process event data
dfevent2 = pd.DataFrame(data = dfcal['Time (s)'], columns = dfcal.columns)
errorindex = []
for i in dfcal.columns[1:]:
    if i in dfevent[' Cell Name'].tolist():
        for j in dfevent[dfevent[' Cell Name'] == i].index:
            try:
                indspike = dfcal.index[abs(dfcal['Time (s)'] - dfevent.loc[j,'Time (s)']) <= 0.05].tolist()[0]
                dfevent2.loc[indspike, i] = dfevent.loc[j,' Value']
            except IndexError:
                errorindex.append(j)
CalciumData[day]['Events'] = dfevent2.to_json(orient='columns')

# Processing motion data
# Change the column names in DLC data
for i in dfmotions:
    i.columns = ['Time (s)','neck_x','neck_y','neck_p','left_ear_x','left_ear_y','left_ear_p','right_ear_x','right_ear_y','right_ear_p']
    # Remove unwanted rows and columns
    i.drop(index=[0,1],inplace=True)
    i.reset_index(drop=True,inplace=True)
#%% Extract information from GPIO

# Find behavioural video start and end time
dfgpioDLC = dfgpio.loc[dfgpio[' Channel Name']==' BNC Trigger Input']
latencym = dfgpioDLC.iloc[1]['Time (s)']
vidst=[]
videt=[]
for i in dfgpioDLC.index:
    if dfgpioDLC.loc[i,' Value'] > 0:
        vidst.append(dfgpioDLC.loc[i,'Time (s)'] - latencym)
        videt.append(dfgpioDLC.loc[i+1,'Time (s)'] - latencym)

# Find calcium recording sessions start and end time
dfgpioCal = dfgpio.loc[dfgpio[' Channel Name']==' EX-LED']
latencyc= dfgpioCal.iloc[1]['Time (s)']
calstG=[]
caletG=[]
for i in dfgpioCal.index:
    if dfgpioCal.loc[i][' Value'] > 0:
        calstG.append(dfgpioCal.loc[i,'Time (s)'] - latencyc)
        caletG.append(dfgpioCal.loc[i+1,'Time (s)'] - latencyc)

# Find time points in calcium data that correspond to start and end time
calst = []
calet = []
for i in calstG:
    v1 = min(x for x in dfcal['Time (s)'] if x >= i)
    v2 = max([x for x in dfcal['Time (s)'] if x <= i], default=0.0)
    if abs(v1-i) < abs(v2-i):
        calst.append(v1)
    else:
        calst.append(v2)
for i in caletG:
    v1 = min([x for x in dfcal['Time (s)'] if x >= i], default=9999)
    v2 = max(x for x in dfcal['Time (s)'] if x <= i)
    if abs(v1-i) < abs(v2-i):
        calet.append(v1)
    else:
        calet.append(v2)

# Find index numbers corresponding to start and end time
indst = []
indet = []
for i in calst:
    inds = dfcal.index[dfcal['Time (s)'] == i].tolist()[0]
    indst.append(inds)
for i in calet:
    inde = dfcal.index[dfcal['Time (s)'] == i].tolist()[0]
    indet.append(inde)
CalciumData[day]['indst']=indst
CalciumData[day]['indet']=indet

# Store calcium data from each session in separate dataframes
sessions = {'session'+str(i):dfcal[indst[i]:indet[i]+1].copy() for i in range(len(calst))}
for n, i in sessions.items():
    i.iloc[:,1:] = i.iloc[:,1:]/(i.iloc[:,1:].max()) # Normalize activity
    i['Population Data'] = i.iloc[:,1:].sum(axis=1)/i.iloc[:,1:].shape[1] # Calculate average
    i.reset_index(drop=True,inplace=True)
    i['Time (s)'] -= i['Time (s)'][0]
    i=i.to_json(orient='columns')
CalciumData[day]['Traces']=sessions
#%% Interpolation to get position and speed, using timestamp from calcium data

pthresh = 0.99 # Threshold for motion detection
px2cm = 490/30 # converting pixel to cm
labeled_frames = []
for s in range(len(dfmotions)):
    currsession = list(CalciumData[day]['Traces'].values())[s].copy()
    dfmotion = dfmotions[s].copy()
    dfmotion = dfmotion.reset_index(drop=True)
    dfmotion = dfmotion.astype(float)
    # Choose between midpoint or neck as marker, filter based on probability
    dfmotion['head_x']=np.nan
    dfmotion['head_x'].loc[(dfmotion['neck_p']>=pthresh)] = dfmotion['neck_x']
    dfmotion['head_x'].loc[(dfmotion['left_ear_p']>=pthresh) & (dfmotion['right_ear_p']>=pthresh)] = (dfmotion['left_ear_x']+dfmotion['right_ear_x'])/2
    dfmotion['head_y']=np.nan
    dfmotion['head_y'].loc[(dfmotion['neck_p']>=pthresh)] = dfmotion['neck_y']
    dfmotion['head_y'].loc[(dfmotion['left_ear_p']>=pthresh) & (dfmotion['right_ear_p']>=pthresh)] = (dfmotion['left_ear_y']+dfmotion['right_ear_y'])/2
    # Save numbers of labeled frames
    labeled_frames.append(dfmotion['head_x'].count())
    # Identify gaps, interpolate
    dfmotion['head_x'].interpolate(inplace=True)
    dfmotion['head_y'].interpolate(inplace=True)
    # Interpolate datapoints in dfmotion to match dfcal
    newt = list(np.linspace(dfmotion['Time (s)'].iloc[0],dfmotion['Time (s)'].iloc[-1],indet[s]-indst[s]+1))
    currsession['newx'] = np.interp(newt,dfmotion['Time (s)'],dfmotion['head_x'])
    currsession['newy'] = np.interp(newt,dfmotion['Time (s)'],dfmotion['head_y'])
    # Calculate average position
    currsession['x']=currsession['newx'].rolling(window=5,center=True).mean()
    currsession['y']=currsession['newy'].rolling(window=5,center=True).mean()
    # Calculate speed
    spd = [np.nan]
    for i in currsession.index[:-1]:
        speed = (((currsession.loc[i+1,'x']-currsession.loc[i,'x'])**2 + (currsession.loc[i+1,'y']-currsession.loc[i,'y'])**2)**0.5)/px2cm*10
        spd.append(speed)
    currsession['Speed']=spd
    # Update generated data in stored data
    CalciumData[day]['Traces']['session' + str(s)] = currsession.to_json(orient='columns')
CalciumData[day]['Labeled frames'] = labeled_frames
#%%
'''
Input/output
'''
#%% Read cell maps from JSON
'''
Manual input: indicate filename of CellMaps file
'''
plotday='0715'
with open(plotday + ' CellMaps.json', 'r') as f:
    CellMaps = json.load(f)
#%% Export generated maps into JSON
with open('ProcessedData.json', 'w', encoding='utf-8') as f:
    json.dump(CalciumData, f, ensure_ascii=False, indent=4)
#%%
with open('ProcessedData.pkl', 'wb') as f:
    pickle.dump(CalciumData, f)