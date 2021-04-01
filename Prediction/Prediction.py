# %%
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
from download import download
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from pmdarima import auto_arima
from math import ceil,sqrt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# %%
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQVtdpXMHB4g9h75a0jw8CsrqSuQmP5eMIB2adpKR5hkRggwMwzFy5kB-AIThodhVHNLxlZYm8fuoWj/pub?gid=2105854808&single=true&output=csv'
path_target = "Times_Velos.csv"
data_raw = pd.read_csv(url)
data_raw.columns=['Date','Hour','Grand total',"Todaystotal", 'Unnamed','Remark']


# %%
data = data_raw.copy()
data.drop(columns=['Unnamed', 'Remark', 'Grand total'], inplace=True)
data.dropna(inplace = True)
data.info()


# %%
mask = ( data['Hour'] >= '00:01:00') & (data['Hour'] < '09:00:00')
data = data.loc[mask]
data['Date'] = pd.to_datetime(data['Date'], format = '%d/%m/%Y' )
data


# %%
bike = data.groupby(data['Date']).sum()
bike


# %%
filter = bike.index <= '2020-05-11'
bike.drop(index = bike[filter].index, inplace = True )
bike


# %%
bike.plot(figsize=(12, 4))
plt.xlabel('Date')
plt.ylabel('Number of bikes')
plt.title("The daily number of bikes  between midnight and 9AM")
plt.show()


# %%
# Fit auto_arima function to AirPassengers dataset
stepwise_fit = auto_arima(bike, start_p = 1, start_q = 1,
                          max_p = 3, max_q = 3, m = 12,
                          start_P = 0, seasonal = True,
                          d = None, D = 1, trace = True,
                          error_action ='ignore',   
                          suppress_warnings = True,  
                          stepwise = True)           # set to stepwise
stepwise_fit.summary()

# %%
size = int(len(bike) * 0.66)
train = bike.iloc[:size]
test = bike.iloc[size:]


# %%
# Fit a SARIMAX(2, 0, 1)x(2, 1, 0, 12) on the training set 
model = SARIMAX(train['Todaystotal'], 
                order = (2, 0, 1), 
                seasonal_order =(2, 1, 0, 12))
  
result = model.fit()


# %%
# Predictions of ARIMA Model against the test set

start = len(train)
end = len(train) + len(test) - 1
  
predictions = result.predict(start, end,
                             typ = 'levels').rename("Predictions")
predictions.index = test.index
ax = test.plot(figsize = (12, 8))
predictions.plot(ax=ax, legend =True)  


# %%
# Calculate mean squared error
mean_squared_error(test, predictions)

# %%
# Train the model on the full dataset
model = SARIMAX(bike, 
                order = (2, 0, 1), 
                seasonal_order =(2, 1, 0, 12))
result = model.fit()

 # %% 

index_future_dates = pd.date_range(start= '2021-03-31 ', end= '2021-04-20', freq = 'D')
end2 = len(bike) + len(index_future_dates) - 1
forecast = result.predict(start = len(bike), 
                          end = end2, 
                          typ = 'levels').rename('Forecast')
F = pd.DataFrame(forecast)
F.index = index_future_dates
F

# %% 
# Plot the forecast values
ax = bike.plot(figsize = (12, 8))
F.plot(ax=ax)
print(f"the number of bicycle passing between 00:01 AM and 09:00 AM on 2nd April  is {ceil(F['Forecast'][2])}.")






















































































# %%
