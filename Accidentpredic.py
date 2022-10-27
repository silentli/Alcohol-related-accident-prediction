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
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import pickle

#%% bulid AIC Model to test the parameters for SARIMA
def SARIMA_AIC(exog, parameters_list, d, D, s):
    """Return parameters and AIC
        
        parameters_list - list with (p, q, P, Q) 
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    results = []
    
    for param in tqdm_notebook(parameters_list):
        try: 
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        results.append([param, aic])
    
    result_table = pd.DataFrame(results)
    result_table.columns = ['Parameters', 'AIC']
    
    result_table = result_table.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_table


#%% Read Data
data = pd.read_csv('verkehrsunfaelle.csv')


#%% Check & Clean Data
data.info()
data = data[['MONATSZAHL','AUSPRAEGUNG','MONAT','WERT']]

null = data[data.isnull().T.any()]
data = data.dropna()
data = data[~data['MONAT'].isin(['Summe'])]
data = data[data['AUSPRAEGUNG'].isin(['insgesamt'])]
category = data['MONATSZAHL'].unique()

#Create a new variable 'unfall' to save the collated data
unfall = pd.DataFrame()
unfall['MONAT'] = data['MONAT'].unique()
for i in range(len(category)):
    unfall = pd.merge(unfall, data[data['MONATSZAHL'].isin([category[i]])].loc[:,['MONAT','WERT']], how='left', on='MONAT')
    unfall.rename(columns={'WERT':category[i]},inplace=True)

#Add time series index
unfall['MONAT'] = unfall['MONAT'].apply(lambda x : datetime.strptime(x, '%Y%m'))
unfall = unfall.sort_values(by = 'MONAT')
unfall.set_index('MONAT', inplace=True)
unfall.head()

#%% Visualisation of three data sets
unfall.plot(subplots=True, figsize=(10,12))
#plt.show()

#%%Split data of Alkoholunfälle after 2020 
train = unfall['2000':'2020'][category[0]].copy()
test = unfall['2021'][category[0]].copy()
train.plot()
plt.show()

#%%Visualisation of first difference, ACF and PACF
diff1 = train.diff(1).dropna()
diff1.plot()
plt.show()

plot_acf(diff1, lags=60)
plot_pacf(diff1)

print(acorr_ljungbox(diff1.dropna(), lags=12, boxpierce=True,return_df=True))

#%% Apply AIC model, determine parameters
p = range(0,3)
d = 1
q= range(0,3)
P = range(0,3)
D = 1
Q = range(0,2)
s = 12
parameters_list = list(product(p,q,P,Q))
len(parameters_list)

warnings.filterwarnings("ignore") 
AIC_result = SARIMA_AIC(train, parameters_list,d,D,s)

p, q, P, Q = AIC_result['Parameters'][0]

#%%Apply SARIMA, save prediction results
train_model = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,s)).fit(disp=-1)
print(train_model.summary())
train_model.plot_diagnostics(figsize=(12,12))

unfall_alcohol = pd.DataFrame(unfall[category[0]].copy())
unfall_alcohol['SARIMA'] = train_model.fittedvalues.round()
unfall_alcohol['SARIMA'][:s+d] = np.NAN
unfall_alcohol['SARIMA'][test.index[0]:test.index[-1]] = train_model.predict(start=test.index[0], end=test.index[-1]).round()
unfall_alcohol['DEVIATION'] = (unfall_alcohol['SARIMA']-unfall_alcohol[category[0]])

#%% Visualisation of prediction results
plt.figure(figsize=(16,8))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(unfall_alcohol['SARIMA'],color='pink', label='SARIMA')
plt.legend()
plt.show()

print(unfall_alcohol['2021-01'])

#%% 
with open('model.pickle','wb') as file:
    pickle.dump(train_model,file)


#%% 


with open('model.pickle', 'rb') as file:
    model=pickle.load(file)
    print(model.forecast('2021-01')[0].round())
    