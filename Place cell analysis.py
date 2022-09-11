# -*- coding: utf-8 -*-
#%% Importing packages
import pandas as pd
from pathlib import Path  # to work with dir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics as st
from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy import stats
import json
import random
#%% Indicate day for plotting
'''
MANUAL INPUT REQUIRED
Indicate the session to be plotted, run before plotting
'''
plotday = '0805'
plotevent = CalciumData[plotday]['Events']
indst = CalciumData[plotday]['indst']
indet = CalciumData[plotday]['indet']
sessions = [0,1,2,3,4]
'''
MANUAL INPUT REQUIRED
Check labeled frames to get x boundary and grid length
'''
xlens = [490,490,490,490,490]
xbounds = [60,75,60,80,70]

CellMaps = {'day':plotday} # Resets CellMaps that stores all maps for the recording day
#%% Generate occupancy-normalized event-rate maps, 1cm bins
for session in sessions:
    if ('session' + str(session)) not in CellMaps.keys():
        CellMaps['session' + str(session)]={}
    plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
    indmid = int(len(plotsession.index)*0.5)
    xlen = xlens[session]
    xbound = xbounds[session]
    xgridlen=xlen/30
    dfzero = pd.DataFrame(0, columns = [i for i in range(30)],index = [i for i in range(30)])
    dfoc = dfzero.copy()
    for i in plotsession.index: # Calculate occupancy
        xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
        if xgrid < 0 or xgrid > 29: # Skip out of boundary data
            continue
        ygrid = plotsession.loc[i]['y']//16
        if ygrid < 0 or ygrid > 29:
            continue
        try:
            dfoc.at[ygrid,xgrid] += 1
        except KeyError:
            print(i)
    dfoc.replace(0,np.nan,inplace=True) # Ignore unvisited bins
    CellMaps['session' + str(session)]['Occupancys'] = dfoc.to_json(orient='columns')
    
    for cell in plotevent.columns[1:]: # Calculate event map for each cell
        dfhm = dfzero.copy()
        indeventspre = plotevent[indst[session]:indet[session]+1].loc[plotevent[cell]> 0.0].index.tolist() # Grab index of events
        indevents = [i - indst[session] for i in indeventspre]
        for i in indevents:
            if plotsession['Speed'].loc[i] > 0.5: # Only count when speed > 0.5
                xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
                if xgrid < 0 or xgrid > 29: # Skip out of boundary data
                    continue
                ygrid = plotsession.loc[i]['y']//16
                if ygrid < 0 or ygrid > 29:
                    continue
                dfhm.at[ygrid,xgrid] += plotevent.loc[i+indst[session]][cell]
        dfrt = dfhm/dfoc
        dfrt.replace(np.nan,0,inplace=True)
        dfrtgs=gaussian_filter(dfrt,sigma=2.5,truncate=2.0) # Gaussian smoothing with delta = 2.5 cm, 3*3 window
        dfrtgs=pd.DataFrame(dfrtgs/np.nanmax(dfrtgs.max())) # Normalize
        #dfrtgs.mask(dfrtgs<0.5,0,inplace=True) # Filter out bins with activity < 0.5 peak
        CellMaps['session' + str(session)][cell + 's'] = dfrtgs.to_json(orient='columns') # Save resulting map into JSON string for storage
    
    # Generate occupancy-normalized event-rate maps for 1st and 2nd half
    dfoc = dfzero.copy() # Reset dfoc
    for i in plotsession.index[:indmid]: # Calculate occupancy for 1st session
        xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
        if xgrid < 0 or xgrid > 29: # Skip out of boundary data
            continue
        ygrid = plotsession.loc[i]['y']//16
        if ygrid < 0 or ygrid > 29:
            continue
        try:
            dfoc.at[ygrid,xgrid] += 1
        except KeyError:
            print(i)
    dfoc.replace(0,np.nan,inplace=True) # Ignore unvisited bins
    CellMaps['session' + str(session)]['Occupancys1'] = dfoc.to_json(orient='columns')
    
    for cell in plotevent.columns[1:]: # Calculate event map for each cell
        dfhm = dfzero.copy()
        indeventspre = plotevent[indst[session]:indst[session]+indmid].loc[plotevent[cell]> 0.0].index.tolist()
        indevents = [i - indst[session] for i in indeventspre]
        for i in indevents:
            if plotsession['Speed'].loc[i] > 0.5: # Only count when speed > 0.5
                xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
                if xgrid < 0 or xgrid > 29: # Skip out of boundary data
                    continue
                ygrid = plotsession.loc[i]['y']//16
                if ygrid < 0 or ygrid > 29:
                    continue
                dfhm.at[ygrid,xgrid] += plotevent.loc[i+indst[session]][cell]
        dfrt = dfhm/dfoc
        dfrt.replace(np.nan,0,inplace=True)
        dfrtgs=gaussian_filter(dfrt,sigma = 2.5,truncate=2.0) # Gaussian smoothing with delta = 2.5 cm, 3*3 window
        dfrtgs=pd.DataFrame(dfrtgs/np.nanmax(dfrtgs.max()))
        #dfrtgs.mask(dfrtgs<0.5,0,inplace=True) # Filter out bins with activity < 0.5 peak
        CellMaps['session' + str(session)][cell + 's1'] = dfrtgs.to_json(orient='columns')
        
    dfoc = dfzero.copy() # Reset dfoc
    for i in plotsession.index[indmid:]: # Calculate occupancy for 2nd session
        xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
        if xgrid < 0 or xgrid > 29: # Skip out of boundary data
            continue
        ygrid = plotsession.loc[i]['y']//16
        if ygrid < 0 or ygrid > 29:
            continue
        try:
            dfoc.at[ygrid,xgrid] += 1
        except KeyError:
            print(i)
    dfoc.replace(0,np.nan,inplace=True) # Ignore unvisited bins
    CellMaps['session' + str(session)]['Occupancys2'] = dfoc.to_json(orient='columns')
    
    for cell in plotevent.columns[1:]: # Calculate event map for each cell
        dfhm = dfzero.copy()
        indeventspre = plotevent[indst[session]+indmid:indet[session]+1].loc[plotevent[cell]> 0.0].index.tolist()
        indevents = [i - indst[session] for i in indeventspre]
        for i in indevents:
            if plotsession['Speed'].loc[i] > 0.5: # Only count when speed > 0.5
                xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
                if xgrid < 0 or xgrid > 29: # Skip out of boundary data
                    continue
                ygrid = plotsession.loc[i]['y']//16
                if ygrid < 0 or ygrid > 29:
                    continue
                dfhm.at[ygrid,xgrid] += plotevent.loc[i+indst[session]][cell]
        dfrt = dfhm/dfoc
        dfrt.replace(np.nan,0,inplace=True)
        dfrtgs=gaussian_filter(dfrt,sigma = 2.5,truncate=2.0) # Gaussian smoothing with delta = 2.5 cm, 3*3 window
        dfrtgs=pd.DataFrame(dfrtgs/np.nanmax(dfrtgs.max()))
        #dfrtgs.mask(dfrtgs<0.5,0,inplace=True) # Filter out bins with activity < 0.5 peak
        CellMaps['session' + str(session)][cell + 's2'] = dfrtgs.to_json(orient='columns')
    
    # Generate raw event-rate maps, 3 cm bins
    dfzero = pd.DataFrame(0, columns = [i for i in range(10)],index = [i for i in range(10)])
    dfoc = dfzero.copy()
    xgridlen=xlen/10
    for i in plotsession.index:
        #if currsession['Speed'].loc[i] > 0.5: # Only count when speed > 0.5
        xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
        if xgrid < 0 or xgrid > 9: # Skip out of boundary data
            continue
        ygrid = plotsession.loc[i]['y']//48
        if ygrid < 0 or ygrid > 9:
            continue
        try:
            dfoc.at[ygrid,xgrid] += 1
        except KeyError:
            print(i)
    dfoc.replace(0,np.nan,inplace=True) # Ignore unvisited bins
    CellMaps['session' + str(session)]['Occupancyr'] = dfoc.to_json(orient='columns')
    
    for cell in plotevent.columns[1:]:
        dfhm = dfzero.copy()
        indeventspre = plotevent[indst[session]:indet[session]+1].loc[plotevent[cell]> 0.0].index.tolist()
        indevents = [i - indst[session] for i in indeventspre]
        for i in indevents:
            if plotsession['Speed'].loc[i] > 0.5: # Only count when speed > 0.5
                xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
                if xgrid < 0 or xgrid > 9: # Skip out of boundary data
                    continue
                ygrid = plotsession.loc[i]['y']//48
                if ygrid < 0 or ygrid > 9:
                    continue
                dfhm.at[ygrid,xgrid] += plotevent.loc[i+indst[session]][cell]
        dfrt = dfhm/dfoc
        CellMaps['session' + str(session)][cell + 'r'] = dfrt.to_json(orient='columns')
#%%
'''
Input/output
'''
#%% Read cell maps from JSON
'''
Manual input: indicate filename of CellMaps file
'''
plotday='0630'
with open(plotday + ' CellMaps.json', 'r') as f:
    CellMaps = json.load(f)
#%% Export generated maps into JSON
with open(plotday + ' CellMaps.json', 'w', encoding='utf-8') as f:
    json.dump(CellMaps, f, ensure_ascii=False, indent=4)
#%%
'''
Plotting
'''
#%% Plot maps from stored values
def plotmap(s,m): # s:session, e.g. 'session0'; m:map, e.g. ' C00s'
    plt.figure()
    ax = plt.subplot(111)
    dfplot = pd.read_json(CellMaps[s][m])
    sns.heatmap(dfplot, square=True, cmap='plasma', vmin=0, vmax=1).set(title= CellMaps['day'] + ' ' + m + ' ' + s)
    ax.axis('off')
    plt.savefig(CellMaps['day'] + ' ' + m + ' ' + s + '.png')
#%% Plot all maps in one day
for s in [k for k in CellMaps.keys() if k != 'day']:
    for m in CellMaps[s].keys():
        #if m.endswith('r'): # s:smoothed, s1:smoothed 1st, s2:smoothed 2nd, r:raw
            plotmap(s,m)
#%% Place cell identification with stability method
dfcorr=pd.DataFrame(columns=[i[:4] for i in CellMaps['session0'].keys() if i.endswith('s1') and i!='Occupancys1'])
dfcorr['row']=[]
for n,s in enumerate([k for k in CellMaps.keys() if k != 'day']):
    dfcorr.loc[n*3,'row']='Threshold '+s
    dfcorr.loc[n*3+1,'row']='Correlation '+s
    dfcorr.loc[n*3+2,'row']='Status '+s
    maps1=[i for i in CellMaps[s].keys() if i.endswith('s1') and i!='Occupancys1']
    maps2=[i for i in CellMaps[s].keys() if i.endswith('s2') and i!='Occupancys2']
    for k in maps1:
        corrlist=[]
        map1 = pd.read_json(CellMaps[s][k]).to_numpy().flatten()
        for i in range(500):
            k2 = maps2[random.randint(0,len(maps2)-1)]
            map2 = pd.read_json(CellMaps[s][k2]).to_numpy().flatten()
            corrlist.append(np.corrcoef(map1,map2)[0][1])
        p95 = np.nanpercentile(corrlist,95)
        dfcorr.loc[n*3,k[:4]]=p95
        k2 = k[:-1]+'2'
        map2 = pd.read_json(CellMaps[s][k2]).to_numpy().flatten()
        dfcorr.loc[n*3+1,k[:4]]=np.corrcoef(map1,map2)[0][1]
        if np.corrcoef(map1,map2)[0][1]>p95:
            dfcorr.loc[n*3+2,k[:4]]='Accepted'
        else:
            dfcorr.loc[n*3+2,k[:4]]='Rejected'
dfcorr.set_index('row',inplace=True)
dfcorr.to_csv(CellMaps['day']+' Place Cell Identification.csv')
#%% Calculate similarity between sessions
plotday='0707'
with open(plotday + ' CellMaps.json', 'r') as f:
    CellMaps = json.load(f)
'''
Manual input: indicate filename of Place Cell Identification
'''
dfcorr=pd.read_csv(plotday + ' Place Cell Identification.csv')
dfcorr.set_index('row',inplace=True)
PlaceCellList=[]
'''
Manual input: indicate the session to be used for place cell identification
'''
for cell in dfcorr.columns:
    if dfcorr.loc['Status session2',cell]=='Accepted': #or dfcorr.loc['Status session3',cell]=='Accepted':
        PlaceCellList.append(cell)
    
dfsim=pd.DataFrame(columns=PlaceCellList)


'''
Manual input: indicate the sessions to be compared
'''
s1='session1'
s2='session2'
maps1=[i for i in CellMaps[s1].keys() if i.endswith('s') and i!='Occupancys']
maps2=[i for i in CellMaps[s2].keys() if i.endswith('s') and i!='Occupancys']
for cell in PlaceCellList:
    simlist=[]
    map1 = pd.read_json(CellMaps[s1][cell +'s']).to_numpy().flatten()
    map2 = pd.read_json(CellMaps[s2][cell +'s']).to_numpy().flatten()
    if np.isnan(np.corrcoef(map1,map2)[0][1]):
        dfsim.drop(columns=cell,inplace=True)
    else:
        dfsim.loc[0,cell]=np.corrcoef(map1,map2)[0][1]
dfsim.to_csv(CellMaps['day']+' Similarity ' + s1 + s2 + '.csv')
#%% T test of similarity
plotday='0707'
dfsim1=pd.read_csv(plotday + ' Similarity session0session2.csv')
dfsim1.set_index('Unnamed: 0',inplace=True)
dfsim2=pd.read_csv(plotday + ' Similarity session1session2.csv')
dfsim2.set_index('Unnamed: 0',inplace=True)

dfcompare=pd.DataFrame(columns=set(list(dfsim1.columns)+list(dfsim2.columns)))
dfcompare.loc['A']=dfsim1.loc[0]
dfcompare.loc['B']=dfsim2.loc[0]
for cell in dfcompare.columns:
    if dfcompare[cell].isnull().values.any():
        dfcompare.drop(columns=cell,inplace=True)
print(stats.ttest_rel(dfcompare.loc['A'],dfcompare.loc['B']))
#%% T test of similarity, circular vs square
plotday='0707'
dfsim1=pd.read_csv(plotday + ' Similarity session0session4.csv')
dfsim1.set_index('Unnamed: 0',inplace=True)
dfsim2=pd.read_csv(plotday + ' Similarity session1session4.csv')
dfsim2.set_index('Unnamed: 0',inplace=True)
dfsim3=pd.read_csv(plotday + ' Similarity session2session4.csv')
dfsim3.set_index('Unnamed: 0',inplace=True)
dfsim4=pd.read_csv(plotday + ' Similarity session3session4.csv')
dfsim4.set_index('Unnamed: 0',inplace=True)

dfcompare=pd.DataFrame(columns=set(list(dfsim1.columns)+list(dfsim2.columns)+list(dfsim3.columns)+list(dfsim4.columns)))
for cell in set(list(dfsim1.columns)+list(dfsim3.columns)):
    if cell in dfsim1.columns and cell in dfsim3.columns:
        dfcompare.loc['Square',cell]=(dfsim1.loc[0,cell]+dfsim3.loc[0,cell])/2
    elif cell in dfsim1.columns:
        dfcompare.loc['Square',cell]=dfsim1.loc[0,cell]
    else:
        dfcompare.loc['Square',cell]=dfsim3.loc[0,cell]
for cell in set(list(dfsim2.columns)+list(dfsim4.columns)):
    if cell in dfsim2.columns and cell in dfsim4.columns:
        dfcompare.loc['Circular',cell]=(dfsim2.loc[0,cell]+dfsim4.loc[0,cell])/2
    elif cell in dfsim2.columns:
        dfcompare.loc['Circular',cell]=dfsim2.loc[0,cell]
    else:
        dfcompare.loc['Circular',cell]=dfsim4.loc[0,cell]
for cell in dfcompare.columns:
    if dfcompare[cell].isnull().values.any():
        dfcompare.drop(columns=cell,inplace=True)

TTestStats,TTestP=stats.ttest_rel(dfcompare.loc['Square'],dfcompare.loc['Circular'])
#%% T test of similarity, visual cue rotation
plotday='0805'
dfsim1=pd.read_csv(plotday + ' Rotation session0session3.csv')
dfsim1.set_index('Unnamed: 0',inplace=True)
dfsim2=pd.read_csv(plotday + ' Rotation session0session3.csv')
dfsim2.set_index('Unnamed: 0',inplace=True)

dfcompare=pd.DataFrame(columns=set(list(dfsim1.columns)+list(dfsim2.columns)))
dfcompare.loc['Arena rotation']=dfsim1.loc['0']
dfcompare.loc['Cue rotation']=dfsim2.loc['0']
for cell in dfcompare.columns:
    if dfcompare[cell].isnull().values.any():
        dfcompare.drop(columns=cell,inplace=True)
TTestStats,TTestP=stats.ttest_rel(dfcompare.loc['Arena rotation'],dfcompare.loc['Cue rotation'])
#%% One way ANOVA of similarity
SimLight=[]
SimShortLight=[]
SimDark=[]
for i in ['C:/Users/brian/.SunLab/session-IBN20220722/0722 Similarity session0session2.csv',
          'C:/Users/brian/.SunLab/session-IBN20220729/0729 Similarity session0session2.csv',
          'C:/Users/brian/.SunLab/session-IBN20220805/0805 Similarity session0session2.csv']:
    dfsim=pd.read_csv(i)
    dfsim.set_index('Unnamed: 0',inplace=True)
    SimLight+=list(dfsim.loc[0].values)
for i in ['C:/Users/brian/.SunLab/session-IBN20220729/0729 Similarity session2session3.csv',
          'C:/Users/brian/.SunLab/session-IBN20220805/0805 Similarity session1session3.csv']:
    dfsim=pd.read_csv(i)
    dfsim.set_index('Unnamed: 0',inplace=True)
    SimShortLight+=list(dfsim.loc[0].values)
for i in ['C:/Users/brian/.SunLab/session-IBN20220805/0805 Similarity session2session4.csv']:
    dfsim=pd.read_csv(i)
    dfsim.set_index('Unnamed: 0',inplace=True)
    SimDark+=list(dfsim.loc[0].values)
    
stats.f_oneway(SimLight,SimShortLight,SimDark)
#%% T test of similarity, in two arenas
plotday='0707'
dfsim1=pd.read_csv(plotday + ' Similarity session0session2.csv')
dfsim1.set_index('Unnamed: 0',inplace=True)
dfsim2=pd.read_csv(plotday + ' Similarity session1session3.csv')
dfsim2.set_index('Unnamed: 0',inplace=True)
dfsim3=pd.read_csv(plotday + ' Similarity session0session1.csv')
dfsim3.set_index('Unnamed: 0',inplace=True)
dfsim4=pd.read_csv(plotday + ' Similarity session2session3.csv')
dfsim4.set_index('Unnamed: 0',inplace=True)

dfcompare=pd.DataFrame(columns=dfsim4.columns)
for cell in set(list(dfsim1.columns)+list(dfsim2.columns)):
    if cell in dfsim2.columns and cell in dfsim1.columns:
        dfcompare.loc['Same',cell]=(dfsim1.loc[0,cell]+dfsim2.loc[0,cell])/2
    elif cell in dfsim1.columns:
        dfcompare.loc['Same',cell]=dfsim1.loc[0,cell]
    else:
        dfcompare.loc['Same',cell]=dfsim2.loc[0,cell]
for cell in set(list(dfsim3.columns)+list(dfsim4.columns)):
    if cell in dfsim3.columns and cell in dfsim4.columns:
        dfcompare.loc['Different',cell]=(dfsim3.loc[0,cell]+dfsim4.loc[0,cell])/2
    elif cell in dfsim3.columns:
        dfcompare.loc['Different',cell]=dfsim3.loc[0,cell]
    else:
        dfcompare.loc['Different',cell]=dfsim4.loc[0,cell]
for cell in dfcompare.columns:
    if dfcompare[cell].isnull().values.any():
        dfcompare.drop(columns=cell,inplace=True)
TTestStats,TTestP=stats.ttest_rel(dfcompare.loc['Same'],dfcompare.loc['Different'])
#%% Plot similarity bar plot
plt.figure()
ax=sns.swarmplot(data=dfcompare.T,color=".25").set(ylim=(-1,1),title='Similarity Score')
ax=sns.boxplot(data=dfcompare.T,showfliers=False,width=0.2)
plt.savefig('Short light reform square vs circular.png', bbox_inches='tight')
#%% Rotation analysis
plotday='0805'
with open(plotday + ' CellMaps.json', 'r') as f:
    CellMaps = json.load(f)
'''
Manual input: indicate filename of Place Cell Identification
'''
dfcorr=pd.read_csv(plotday + ' Place Cell Identification.csv')
dfcorr.set_index('row',inplace=True)
PlaceCellList=[]
'''
Manual input: indicate the session to be used for place cell identification
'''
for cell in dfcorr.columns:
    if dfcorr.loc['Status session2',cell]=='Accepted':
        PlaceCellList.append(cell)

dfrotmax=pd.DataFrame(columns=[0,90,180,270],index=['Counts','95th'])
'''
Manual input: indicate the sessions to be compared
'''
s1='session0'
s2='session2'
maps1=[i for i in CellMaps[s1].keys() if i.endswith('s') and i!='Occupancys']
maps2=[i for i in CellMaps[s2].keys() if i.endswith('s') and i!='Occupancys']

# Chance level
Rot0=[]
Rot90=[]
Rot180=[]
Rot270=[]
for i in range(500):
    dfrot=pd.DataFrame(columns=PlaceCellList, index=[0,90,180,270,'Max Rotation'])
    for cell in PlaceCellList:
        map1 = pd.read_json(CellMaps[s1][cell +'s'])
        if map1.notnull().values.any(): # Exclude cells with their 1st map empty
            map1_90 = map1.T.reindex(range(29,-1,-1))
            map1_90.reset_index(inplace=True, drop=True)
            map1_180 = map1_90.T.reindex(range(29,-1,-1))
            map1_180.reset_index(inplace=True, drop=True)
            map1_270 = map1_180.T.reindex(range(29,-1,-1))
            map1_270.reset_index(inplace=True, drop=True)    
            k2 = maps2[random.randint(0,len(maps2)-1)]
            map2 = pd.read_json(CellMaps[s2][k2])
            # Rotate map2 by corresponding angles
            dfrot.loc[0,cell]=np.corrcoef(map1.to_numpy().flatten(),map2.to_numpy().flatten())[0][1]
            dfrot.loc[90,cell]=np.corrcoef(map1_90.to_numpy().flatten(),map2.to_numpy().flatten())[0][1]
            dfrot.loc[180,cell]=np.corrcoef(map1_180.to_numpy().flatten(),map2.to_numpy().flatten())[0][1]
            dfrot.loc[270,cell]=np.corrcoef(map1_270.to_numpy().flatten(),map2.to_numpy().flatten())[0][1]
            dfrot.loc['Max Rotation',cell]=dfrot[cell].values.argmax(axis=0)
        else:
            dfrot[cell]=np.nan
    Rot0.append((dfrot.loc['Max Rotation']==0).sum())
    Rot90.append((dfrot.loc['Max Rotation']==1).sum())
    Rot180.append((dfrot.loc['Max Rotation']==2).sum())
    Rot270.append((dfrot.loc['Max Rotation']==3).sum())
for n,i in enumerate([Rot0,Rot90,Rot180,Rot270]):
    dfrotmax.iloc[1,n]=np.nanpercentile(i,95)
    
# Calculate actual rotation
dfrot=pd.DataFrame(columns=PlaceCellList, index=[0,90,180,270,'Max Rotation'])
for cell in PlaceCellList: # Calculate correlation with rotations
    map1 = pd.read_json(CellMaps[s1][cell +'s'])
    map2 = pd.read_json(CellMaps[s2][cell +'s'])
    if map1.notnull().values.any() and map2.notnull().values.any(): # Exclude cells with either map empty
        map1_90 = map1.T.reindex(range(29,-1,-1))
        map1_90.reset_index(inplace=True, drop=True)
        map1_180 = map1_90.T.reindex(range(29,-1,-1))
        map1_180.reset_index(inplace=True, drop=True)
        map1_270 = map1_180.T.reindex(range(29,-1,-1))
        map1_270.reset_index(inplace=True, drop=True)      
        dfrot.loc[0,cell]=np.corrcoef(map1.to_numpy().flatten(),map2.to_numpy().flatten())[0][1]
        dfrot.loc[90,cell]=np.corrcoef(map1_90.to_numpy().flatten(),map2.to_numpy().flatten())[0][1]
        dfrot.loc[180,cell]=np.corrcoef(map1_180.to_numpy().flatten(),map2.to_numpy().flatten())[0][1]
        dfrot.loc[270,cell]=np.corrcoef(map1_270.to_numpy().flatten(),map2.to_numpy().flatten())[0][1]
        dfrot.loc['Max Rotation',cell]=dfrot[cell].values.argmax(axis=0)
    else:
        dfrot[cell]=np.nan
dfrotmax.loc['Counts',0]=(dfrot.loc['Max Rotation']==0).sum()
dfrotmax.loc['Counts',90]=(dfrot.loc['Max Rotation']==1).sum()
dfrotmax.loc['Counts',180]=(dfrot.loc['Max Rotation']==2).sum()
dfrotmax.loc['Counts',270]=(dfrot.loc['Max Rotation']==3).sum()

dfrot.to_csv(CellMaps['day']+' Rotation ' + s1 + s2 + '.csv')
dfrotmax.to_csv(CellMaps['day']+' Rotation Count ' + s1 + s2 + '.csv')
#%% Plot distribution of optimal rotation
plotday='0707'
dfrotmax=pd.read_csv(plotday + ' Rotation Count session0session4.csv')
dfrotmax.set_index('Unnamed: 0',inplace=True)
plt.rcParams['font.size'] = '16'
plt.figure(figsize=(5, 5))
#sns.barplot(data=dfrotmax.T, x=['0','90','180','270'],y='Counts',height=4, aspect=.7).set(xlabel='Rotation')
#sns.catplot(data=dfrotmax, kind="bar",ci=None,height=4, aspect=1).set(xlabel='Rotation')
plt.bar(x=['0','90','180','270'],height=dfrotmax.T['Counts'],width=0.5,edgecolor='k',linewidth=2)
plt.xlabel('Rotation',fontsize=20)
plt.ylabel('Count',fontsize=20)
#plt.plot(['0','90','180','270'],dfrotmax.T['Mean'],'k-')
plt.plot(['0','90','180','270'],dfrotmax.T['95th'],'k--')
#plt.plot(['0','90','180','270'],dfrotmax.T['5th'],'k--')
#plt.axvline(x='90', color='r',linestyle='--')
plt.savefig(plotday + ' Rotation Count session0session4.png', bbox_inches='tight')
#%% Distinction index
'''
Manual input
'''
plotday = '0707'
plotevent = CalciumData[plotday]['Events']
indst = CalciumData[plotday]['indst']
indet = CalciumData[plotday]['indet']
sessions = [[0,2],[1,3]]

'''
Manual input: indicate filename of Place Cell Identification
'''
dfcorr=pd.read_csv('0707 Place Cell Identification.csv')
dfcorr.set_index('row',inplace=True)
PlaceCellList=[]
'''
Manual input: indicate the session to be used for place cell identification
'''
for cell in dfcorr.columns:
    if dfcorr.loc['Status session2',cell]=='Accepted' or dfcorr.loc['Status session3',cell]=='Accepted':
        PlaceCellList.append(cell)

# Calculate occupancy in each arena
occupancy=[0,0]
for n, arena in enumerate(sessions):
    for session in arena:
        plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
        occupancy[n]+=plotsession['Speed'].loc[plotsession['Speed']>0.5].size

dfDI=pd.DataFrame(columns=PlaceCellList,index=['DI'])
# Add events
for cell in PlaceCellList: # Calculate event map for each cell
    events=[0,0]
    for n, arena in enumerate(sessions):
        for session in arena:
            plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
            indeventspre = indeventspre = plotevent[indst[session]:indet[session]+1].loc[plotevent[cell]> 0.0].index.tolist()
            indevents = [i - indst[session] for i in indeventspre]
            for i in indevents:
                if plotsession['Speed'].loc[i] > 0.5: # Only count when speed > 0.5
                    events[n]+=plotevent.loc[i+indst[session]][cell]
    ER1=events[0]/occupancy[0]
    ER2=events[1]/occupancy[1]
    dfDI.loc['DI',cell]=(ER1-ER2)/(ER1+ER2)
dfDI.to_csv(plotday +' Distinction Index.csv')
#%% Plot distribution of DI
plotday='0707'
dfDI=pd.read_csv(plotday+' Distinction Index.csv')
dfDI.set_index('Unnamed: 0',inplace=True)
fig=plt.figure()
sns.histplot(data=dfDI.T,bins=20,binrange=(-1,1),legend=False).set(xlabel='Distinction Index')
plt.savefig('DI.png', bbox_inches='tight')
#%% Plot ratemap across sessions
'''
'''
dfzero = pd.DataFrame(0, columns = [i for i in range(30)],index = [i for i in range(30)])
sessions=[0,1,2,3]
xlens = [490,470,490,460]
xbounds = [75,95,70,100]
CellMaps = {'day':plotday}

for session in sessions:
    if ('session' + str(session)) not in CellMaps.keys():
        CellMaps['session' + str(session)]={}
    plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
    xlen = xlens[session]
    xbound = xbounds[session]
    xgridlen=xlen/30
    dfoc = dfzero.copy()
    for i in plotsession.index: # Calculate occupancy
        xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
        if xgrid < 0 or xgrid > 29: # Skip out of boundary data
            continue
        ygrid = plotsession.loc[i]['y']//16
        if ygrid < 0 or ygrid > 29:
            continue
        try:
            dfoc.at[ygrid,xgrid] += 1
        except KeyError:
            print(i)
    dfoc.replace(0,np.nan,inplace=True) # Ignore unvisited bins
    CellMaps['session' + str(session)]['Occupancys'] = dfoc.to_json(orient='columns')
    
for cell in PlaceCellList: # Calculate event map for each cell
    MaxRate=0
    for session in sessions:
        plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
        xlen = xlens[session]
        xbound = xbounds[session]
        xgridlen=xlen/30
        dfhm = dfzero.copy()
        dfoc = pd.read_json(CellMaps['session' + str(session)]['Occupancys'])
        indeventspre = plotevent[indst[session]:indet[session]+1].loc[plotevent[cell]> 0.0].index.tolist() # Grab index of events
        indevents = [i - indst[session] for i in indeventspre]
        for i in indevents:
            if plotsession['Speed'].loc[i] > 0.5: # Only count when speed > 0.5
                xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
                if xgrid < 0 or xgrid > 29: # Skip out of boundary data
                    continue
                ygrid = plotsession.loc[i]['y']//16
                if ygrid < 0 or ygrid > 29:
                    continue
                dfhm.at[ygrid,xgrid] += plotevent.loc[i+indst[session]][cell]
        dfrt = dfhm/dfoc
        dfrt.replace(np.nan,0,inplace=True)
        dfrtgs=pd.DataFrame(gaussian_filter(dfrt,sigma=2.5,truncate=2.0)) # Gaussian smoothing with delta = 2.5 cm, 3*3 window
        if np.nanmax(dfrtgs.max())>MaxRate:
            MaxRate=np.nanmax(dfrtgs.max())
        CellMaps['session' + str(session)][cell + 's'] = dfrtgs.to_json(orient='columns') # Save resulting map into JSON string for storage
    for session in sessions:
        dfnorm=pd.read_json(CellMaps['session' + str(session)][cell+'s'])
        dfnorm/=MaxRate
        CellMaps['session' + str(session)][cell + 's'] = dfnorm.to_json(orient='columns')