#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 00:43:01 2022

@author: guolee
"""

#%% Import packages
import warnings
import pandas as pd, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import pickle


#%% bulid AIC Model to test the parameters for SARIMA/ARIMA/ETS
def AIC(exog, parameters_list, T):
    """Return parameters and AIC       
        parameters_list - list with (p,d,q,P,D,Q,s) 
        p - Trend autoregression order
        d - Trend difference order
        q - Trend moving average order
        P - Seasonal autoregressive order
        D - Seasonal difference order
        Q - Seasonal moving average order
        s - The number of time steps for a single seasonal period.
    """
    results = []
    #Progress bar for jupyter notebook    
    for param in tqdm_notebook(parameters_list):
        try:
            if T =='SARIMA':
                #order=(p,d,q), seasonal_order=(P,D,Q,s), disp<0 means no output information print
                model = SARIMAX(exog, order=(param[:3]), seasonal_order=(param[3:])).fit(disp=-1)
            elif T =='ARIMA':
                model = SARIMAX(exog, order=param).fit(disp=-1)
            elif T == 'ETS':
                model = ExponentialSmoothing(exog, trend=param[0], seasonal=param[1], seasonal_periods=param[2], damped=param[3]).fit()
        except:
            continue
        aic = model.aic
        results.append([param, aic])
    
    result_table = pd.DataFrame(results)
    result_table.columns = ['Parameters', 'AIC']
    
    #Sorting AIC with the sort_values(), in an ascending order
    result_table = result_table.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_table

#%% Add time features for ML
def create_time_features(data, target=None):
    """
    Creates time series features from datetime index
    """
    df = data.copy()
    df['date'] = df.index
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_day'] = np.sin(df['dayofyear'])
    df['cos_day'] = np.cos(df['dayofyear'])
    X = df.drop(['date'], axis=1)
    if target:
        y = X[target]
        X = X.drop([target], axis=1)
        return X, y

    return X

#%% Read Data
data = pd.read_csv('verkehrsunfaelle.csv')

#%% Check & Clean Data
data.info()
data = data[['MONATSZAHL','AUSPRAEGUNG','MONAT','WERT']]

#isnull() identifies whether the element in the data is a null value
#any() identifies whether the row has a null value (axis=1)
null = data[data.isnull().any(axis=1)]

#Delete rows with null values (axis=0)
data = data.dropna()
# ~ to invert boolean series, isin (identifies row containing 'Summe')
data = data[~data['MONAT'].isin(['Summe'])]
data = data[data['AUSPRAEGUNG'].isin(['insgesamt'])]
#find accident categories
category = data['MONATSZAHL'].unique()

#Create 'unfall' to save the collated data, make it more readable
unfall = pd.DataFrame()
unfall['MONAT'] = data['MONAT'].unique()
for i in range(len(category)):
    #.loc[row,column], merge is like vlookup in Excel
    unfall = pd.merge(unfall, data[data['MONATSZAHL'].isin([category[i]])].loc[:,['MONAT','WERT']], how='left', on='MONAT')
    #rename 'WERT'
    unfall.rename(columns={'WERT':category[i]},inplace=True)

#Add time series index
unfall.rename(columns={'MONAT':'Date'},inplace=True)
#unfall['Date'] = unfall['Date'].apply(lambda x : pd.to_datetime(str(x), format='%Y%m'))
unfall['Date'] = unfall['Date'].apply(lambda x : datetime.strptime(x, '%Y%m'))
#sorting 'Date' in an ascending order
unfall = unfall.sort_values(by = 'Date')
unfall.set_index('Date', inplace=True)
unfall.head()

#%% Visualisation of three data sets
#Assume that traffic accidents do not include alcohol accidents and escape accidents.
#Alcohol Accidents have the lowest numbers and all three sets show some seasonality.
cmap = plt.get_cmap('Set1',3)
unfall.plot(subplots=True, figsize=(10,12), colormap=cmap)
#plt.show()

#Annual Traffic Accidents
#Alcohole Accidents show a decreasing trend.
unfall.resample('Y').sum().plot(subplots=True, figsize=(10,12), xlabel='Year',colormap=cmap, style='.-')
plt.show()

#Average percentage of traffic accidents in Pie Chart
#highlights the proportion of accidents
plt.pie(
        unfall.mean(),
        labels=[category[0],category[1],category[2]],
        colors= cmap.colors,
        explode=(0.5, 0, 0),
        autopct='%.2f%%'
        )
plt.title('Average proportion of traffic accidents 2000-2021')
plt.show()

#Monthly accidents as a proportion of the year (Average)
#High correlation between escape accidents and traffic accidents.
#July, September and October are the peak months of the year.
unfall_month= unfall.set_index(unfall.index.month) #DatetimeIndex
#unfall_month=100*unfall_month.mean(level=0)/(12*unfall_month.mean())
unfall_month=100*unfall_month.groupby('Date').agg('mean')/(12*unfall_month.mean())

unfall_month.plot(
                  colormap=cmap, style='.-',
                  title='Monthly accidents as a proportion of the year (Average)',
                  xlabel='Month',ylabel='%',
                  figsize=(10,6)
                  )

#%%Split data of Alkoholunfälle after 2020 
train = unfall.loc['2000':'2020',[category[0]]].copy()
test = unfall.loc['2021',[category[0]]].copy()
train.plot()
#plt.show()

#To see if it has seasonal characteristic
train['2002':'2004'].plot()
#plt.show()
train['2012':'2015'].plot()
#plt.show()

#%%Stationary test (Augmented Dickey-Fuller Test)
#Calculate first differencing of the series
diff1 = train.diff(1).dropna()
diff1.plot(title='First-difference of training set')
#plt.show()

#took the difference over a period of 12 months
#shows that the order of seasonal differencing (D) is 1
diff1_seasonal = diff1.diff(12).dropna()
diff1_seasonal.plot(title='Seasonal differencing')
#plt.show()

print("p-value: %f" %adfuller(train)[1])
#diff1 is stationary from the result of Augmented Dickey-Fuller Test
print("Diff1-p-value: %f" %adfuller(diff1)[1])
print("Diff1_seasonal-p-value: %f" %adfuller(diff1_seasonal)[1])


#%%Visualisation of ACF and PACF (with first difference)
#plot_acf(diff1, lags=60)
#plot_pacf(diff1)

#not white noise
print(acorr_ljungbox(diff1_seasonal, lags=12, boxpierce=True,return_df=True))

#ACF shows significant spikes at lag 1 or 2, q could be 1 or 2
#ACF also shows significant spikes at lag 12, Q coulld be 1.
plot_acf(diff1_seasonal, lags=60)
#PACF has a significant peak at lag 0, which suggest AR(0) p=0
#PACF has not significant peak at lag 12/24, which suggest P=0 (a seasonal autoregressive process of order 0)
plot_pacf(diff1_seasonal, lags=60)

#%% DataFrame unfall_alcohol
unfall_alcohol = pd.DataFrame(unfall[category[0]].copy())

#%% Apply AIC model, determine parameters for ARIMA
p_a = range(0,4)
d_a = range(1,2) #which is 1
q_a = range(0,4)
model_type = 'ARIMA'

parameters_arima = list(product(p_a,d_a,q_a))
len(parameters_arima)
warnings.filterwarnings("ignore") 
AIC_arima = AIC(train, parameters_arima, model_type)

p_a, d_a, q_a = AIC_arima['Parameters'][0]

#%%Apply ARIMA, save prediction results
train_arima = SARIMAX(train, order=(p_a,d_a,q_a)).fit(disp=-1)
print(train_arima.summary())
train_arima.plot_diagnostics(figsize=(10,11))

unfall_alcohol['ARIMA'] = train_arima.fittedvalues.round()
unfall_alcohol['ARIMA'][:d_a] = np.NAN
unfall_alcohol['ARIMA'][test.index] = train_arima.predict(start=test.index[0], end=test.index[-1]).round()
unfall_alcohol['D_ARIMA'] = (unfall_alcohol['ARIMA']-unfall_alcohol[category[0]])
rmse_arima = sqrt(mean_squared_error(test, unfall_alcohol['2021']['ARIMA'])) 

#%% Apply AIC model, determine parameters for SARIMA
p_s = range(0,3)
d_s = range(1,2)  #which is 1
q_s = range(0,3)
P_s = range(0,3)
D_s = range(1,2) #which is 1
Q_s = range(0,2)
s_s = range(12,13) #which is 12
model_type = 'SARIMA'
parameters_sarima = list(product(p_s,d_s,q_s,P_s,D_s,Q_s,s_s))
len(parameters_sarima)

warnings.filterwarnings("ignore") 
AIC_sarima = AIC(train, parameters_sarima, model_type)

p_s, d_s, q_s, P_s, D_s, Q_s, s_s = AIC_sarima['Parameters'][0]


#%%Apply SARIMA, save prediction results
train_sarima = SARIMAX(train, order=(p_s,d_s,q_s), seasonal_order=(P_s,D_s,Q_s,s_s)).fit(disp=-1)
print(train_sarima.summary())
#to check if the residuals are white noise
train_sarima.plot_diagnostics(figsize=(10,11))

unfall_alcohol['SARIMA'] = train_sarima.fittedvalues.round()
unfall_alcohol['SARIMA'][:s_s+d_s] = np.NAN
unfall_alcohol['SARIMA'][test.index] = train_sarima.predict(start=test.index[0], end=test.index[-1]).round()
unfall_alcohol['D_SARIMA'] = (unfall_alcohol['SARIMA']-unfall_alcohol[category[0]])
rmse_sarima = sqrt(mean_squared_error(test, unfall_alcohol['2021']['SARIMA'])) 

#%%perform AIC, determine parameters for Exponential Smoothing(ETS)
t_e = ['add', 'mul',None]
s_e = ['add', 'mul',None]
p_e = range(12,13)
d_e = [True, False]
model_type = 'ETS'

parameters_ets = list(product(t_e,s_e,p_e,d_e))

warnings.filterwarnings("ignore") 
AIC_ets = AIC(train, parameters_ets, model_type)

t_e, s_e, p_e, d_e = AIC_ets['Parameters'][0]

#%%perform ETS(Holt-Winter’s Smoothing model)
train_ets = ExponentialSmoothing(train, trend=t_e, seasonal=s_e, seasonal_periods=p_e, damped=d_e).fit()
print(train_ets.summary())

unfall_alcohol['ETS'] = train_ets.fittedvalues.round()
unfall_alcohol['ETS'][test.index] = train_ets.predict(start=test.index[0], end=test.index[-1]).round()
unfall_alcohol['D_ETS'] = (unfall_alcohol['ETS']-unfall_alcohol[category[0]])
rmse_ets = sqrt(mean_squared_error(test, unfall_alcohol['2021']['ETS'])) 

#%% Set data for multivariate time series forecasting
x_train, y_train= create_time_features(train, target=category[0])
x_test, y_test = create_time_features(test, target=category[0])
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#%% Random Forest
reg = RandomForestRegressor(n_estimators = 1000, random_state = 42)
reg.fit(x_train, y_train)
yhat = reg.predict(x_test)
unfall_alcohol['Randomforest'] = np.nan
unfall_alcohol['Randomforest'][test.index] = yhat.round()
rmse_randomforest = sqrt(mean_squared_error(test, unfall_alcohol['2021']['Randomforest'])) 

#%% Visualisation of results
plt.figure(figsize=(16,8))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(unfall_alcohol['SARIMA'],color='pink', label='SARIMA')
plt.legend()
plt.show()


plt.figure(figsize=(16,8))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(unfall_alcohol['ARIMA'],color='pink', label='ARIMA')
plt.legend()
plt.show()

plt.figure(figsize=(16,8))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(unfall_alcohol['ETS'],color='pink', label='ETS')
plt.legend()
plt.show()

plt.figure(figsize=(16,8))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(unfall_alcohol['Randomforest'],color='pink', label='RandomForest')
plt.legend()
plt.show()

print('rmse_sarima: %f\nrmse_arima: %f\nrmse_ets: %f\nrmse_randomforest: %f\n' %(rmse_sarima, rmse_arima, rmse_ets, rmse_randomforest))
print(unfall_alcohol['2021'])


#%% save the model into model.pickle
with open('model.pickle','wb') as file:
    pickle.dump(train_sarima,file)


#%% read the model from model.pickle

with open('model.pickle', 'rb') as file:
    model=pickle.load(file)
