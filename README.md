## Alcohol-related Accident Prediction
By analysing alcohol accident data (Munich) from 2002 to 2020 and forecasting numbers for 2021.

### Analysis and transforms
Stationarity:  
1) AC and PAC plots
2) Dickey-Fuller test

Transform:
1) Difference transform
2) Seasonal difference transform

### Models
1) Autoregressive integraded moving average (ARIMA)
2) Seasonal autoregressive integrated moving average (SARIMA)
3) Exponential Smoothing (ETS) 

### Evaluation Metrics
1) Root Mean Squared Error (RMSE)

### Deployment
1) save & load the predict model using Pickle
2) Deploying the model to Heroku using Flask  
3) Application URL: https://accident-prediction-heroku.herokuapp.com/predict  
4) POST request with a JSON body:

		{
			"year":2020,
			"month":10
		}

5) Notice: Heroku no longer supporting free plans, the link will not be available.

### Data Source:
([Monthly figures for traffic accidents](https://opendata.muenchen.de/dataset/monatszahlen-verkehrsunfaelle/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7))