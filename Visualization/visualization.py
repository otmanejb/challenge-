# %%
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas_alive
from matplotlib import pyplot as plt
from download import download
from datetime import datetime
import json


# %%
dic = {}  
for i in np.arange(1,11):
    dic[i] = pd.read_json(f'bike{i}.json',lines=True)
    dic[i] = dic[i].groupby(by='dateObserved').sum('intensity')
    dic[i].drop(columns=['laneId','reversedLane'], inplace=True)

# %%
for a in np.arange(1,11):
    dic[a]['Date'] = dic[a].index
    for i in np.arange(0,dic[a].shape[0]):
        dic[a]['Date'][i] = dic[a].index[i][0:10]
    dic[a]['Date'] = pd.to_datetime(dic[a]['Date'])
    dic[a].reset_index(inplace=True)
    dic[a].set_index('Date', inplace=True)
    dic[a].drop(columns=['dateObserved'], inplace=True)



# %%
dic[1]['Location'] = 'Tanneurs'
dic[2]['Location'] = 'Beracasa'
dic[3]['Location'] = 'Lodève Celleneuve'
dic[4]['Location'] = 'Lavèrune'
dic[5]['Location'] = 'Vieille poste'
dic[6]['Location'] = 'Delmas 2'
dic[7]['Location'] = 'Delmas 1'
dic[8]['Location'] = 'Gerhardt'
dic[9]['Location'] = 'Lattes 2 '
dic[10]['Location'] = 'Lattes 1 '
data = pd.concat( [dic[1], dic[2], dic[3], dic[4], dic[5], dic[6], dic[7], dic[8], dic[9], dic[10]])
data = data.sort_index(ascending=True)


# %%
data.reset_index(inplace =True)
data= data.pivot(index="Date", columns="Location", values="intensity").fillna(0)
data.head(25)

# %%
data.plot_animated("Bike_traffic.gif", 
                       period_fmt="%Y-%m-%d", fixed_order = True,
                       title=" the intensity of bicycle traffic at several locations in Montpellier",  orientation='h')
                      



# %%
