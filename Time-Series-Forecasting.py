import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import yfinance

raw_data = yfinance.download (tickers = "^GSPC", start = "1994-01-07", 
                              end = "2019-09-01", interval = "1d")

# Create a new dataframe with one column - "spx"
data = pd.DataFrame(columns = ["spx"])
# Copy closing prices of S&P 500 to this new column
data["spx"] = raw_data["Close"]
# Ensure that the dates are ordered in business week fashion (5 days a week)
data = data.asfreq("b")

print("Null values - ",data.spx.isnull().sum())

print("\nStatistical Description of the series - ")
print(data.describe())

data.spx = data.spx.fillna(method='ffill')
print("\nNull values - ",data.spx.isnull().sum())

# Calculating returns and volatility based on previous formulas
data["spx_ret"] = data.spx.pct_change(1).mul(100)
data["spx_vol"] = data.spx_ret.abs()


# Importing Dataset
data = pd.read_csv("data.csv")
data.Date = pd.to_datetime(data.Date)
data.set_index("Date", inplace = True)

# Displaying first 5 rows of "data" DataFrame
data.head()

# Setting the figure size 
plt.rcParams["figure.figsize"] = (18, 5)

# Subdividing the figure into 3 figures stacked in 1 row
fig, ax = plt.subplots(1, 3)

# First Plot - S&P 500 prices/ against time
ax[0].plot(data.spx, color = "blue", label = "SPX")
ax[0].set_title("SPX Prices", size = 24)
ax[0].legend()

# Second Plot - S&P 500 returns against time
ax[1].plot(data.spx_ret, color = "blue", label = "SPX Returns")
ax[1].set_title("SPX Returns", size = 24)
ax[1].legend()

# Third Plot - S&P 500 volatility against time
ax[2].plot(data.spx_vol, color = "blue", label = "SPX Volatility")
ax[2].set_title("SPX Volatility", size = 24)
ax[2].legend()

# Used to display the plot free from any additional text in Jupyter notebooks
plt.show()

train_df = data.loc[:"2018-12-31"]
test_df = data.loc["2019-01-01":]

print("Training Set Shape - ", train_df.shape)
print("Testing Set Shape - ", test_df.shape)

# Adding another column in train_df storing the "Year" of each observation
train_df["Year"] = train_df.index.year

# Setting the size of the figure 
plt.rcParams["figure.figsize"] = 24, 21

# Defining 3 subplots
fig, axes = plt.subplots(3, 1)

# First Boxplot: Yearly S&P 500 Prices
train_df.boxplot(by ='Year', column =['spx'], ax = axes[0])
axes[0].set_title("SPX Prices", size = 24)

# Second Boxplot: Yearly S&P 500 Returns
train_df.boxplot(by ='Year', column =['spx_ret'], ax = axes[1])
axes[1].set_title("SPX Returns", size = 24)

# Third Boxplot: Yearly S&P 500 Volatility
train_df.boxplot(by ='Year', column =['spx_vol'], ax = axes[2])
axes[2].set_title("SPX Volatility", size = 24)

# Displaying plots
plt.show()

# Setting the figure size
plt.rcParams["figure.figsize"] = 24, 18

# Defining 3 subplots one below the other
fig, axes = plt.subplots(3, 1)

# Plotting the distributions in the respective subplots
sns.distplot(train_df.spx, ax = axes[0])
sns.distplot(train_df.spx_ret, ax = axes[1])
sns.distplot(train_df.spx_vol, ax = axes[2])

# Setting the title for each subplot
axes[0].set_title("SPX Prices", size = 24)
axes[1].set_title("SPX Returns", size = 24)
axes[2].set_title("SPX Volatility", size = 24)

# Displaying the plot
plt.show()

# Import the required package
from statsmodels.tsa.seasonal import seasonal_decompose

# Set Plot size
plt.rcParams["figure.figsize"] = 18, 20

# Call the seasonal_decompose method to decompose the data using the "additive" model 
result = seasonal_decompose(train_df.spx, model='additive')

# Plot the result
result.plot()

# Display the plot
plt.show()

result.seasonal[:20].plot(marker = "o")
plt.show()

train_df = data.loc[:"2018-12-31"]
test_df = data.loc["2019-01-01":]

print("Training Set Shape - ", train_df.shape)
print("Testing Set Shape - ", test_df.shape)

# Importing the necessary package
from statsmodels.tsa.stattools import adfuller

# ADF test in S&P 500 Returns
adfuller(train_df["spx_ret"][1:])

# Importing Required Package
import statsmodels.graphics.tsaplots as sgt
# Fixing plot size
plt.rcParams["figure.figsize"] = 18, 5

# Defining Subplots
fig, axes = plt.subplots(1, 2)

# Plotting ACF and PACF for S&P 500 Returns
sgt.plot_acf(train_df.spx_ret[1:], zero = False, lags = 40, ax = axes[0])
sgt.plot_pacf(train_df.spx_ret[1:], zero = False, lags = 40, ax = axes[1])

# Display the Plot
plt.show()

# MODEL FITTING
# Importing Required Package
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Defining the Model
model = SARIMAX(train_df["spx_ret"][1:], order = (1, 0, 1))
# Fitting the Model
model_results = model.fit()

# Printing the model summary
print(model_results.summary())

# EVALUATING RESIDUALS
# Defining the figure size
plt.rcParams["figure.figsize"] = 18, 5

# Defining the subplots 
fig, axes = plt.subplots(1, 2)

# ACF and PACF for residuals of ARIMA(1, 0, 1)
sgt.plot_acf(model_results.resid[1:], zero = False, lags = 40, ax = axes[0])
sgt.plot_pacf(model_results.resid[1:], zero = False, lags = 40, ax = axes[1])

# Displaying the plots
plt.show()

# MODEL FITTING
# Importing Required Package
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Defining the Model
seas_model = SARIMAX(train_df["spx_ret"][1:], order = (1, 0, 1), seasonal_order = (1, 0, 1, 5))
# Fitting the Model
seas_model_results = seas_model.fit()

# Printing the model summary
print(seas_model_results.summary())


# EVALUATING RESIDUALS
# Defining the figure size
plt.rcParams["figure.figsize"] = 18, 5

# Defining the subplots 
fig, axes = plt.subplots(1, 2)

# ACF and PACF for residuals of SARIMA(1, 0, 1)(1, 0, 1, 5)
sgt.plot_acf(seas_model_results.resid[1:], zero = False, lags = 40, ax = axes[0])
sgt.plot_pacf(seas_model_results.resid[1:], zero = False, lags = 40, ax = axes[1])

# Displaying the plots
plt.show()


# Importing Required Packages
from sklearn.metrics import mean_squared_error

# GENERATING FORECASTS AND CALCULATING ACCURACY
# Forecasts of ARIMA model
pred = model_results.predict(start = test_df.index[0], end = test_df.index[-1])
# Forecasts of SARIMA model
seas_pred = seas_model_results.predict(start = test_df.index[0], end = test_df.index[-1])

# RMSE of ARIMA model
arima_rmse = np.sqrt(mean_squared_error(y_true = test_df["spx_ret"].values, y_pred = pred.values))
# RMSE of SARIMA model
sarima_rmse = np.sqrt(mean_squared_error(y_true = test_df["spx_ret"].values, y_pred = seas_pred.values))


# FORECAST vs ACTUALS PLOT
# Setting the size of the figure
plt.rcParams["figure.figsize"] = 18, 5

# Defining the subplots
fig, ax = plt.subplots(1, 2)

# Actuals vs Predictions for ARIMA(1, 0, 1)
ax[0].plot(test_df["spx_ret"], color = "blue", label = "Actuals")
ax[0].plot(pred, color = "red", label = "ARIMA(1, 0, 1) Predictions")
ax[0].set_title(f"ARIMA(1, 0, 1) Predictions (RMSE: {np.round(arima_rmse, 3)})", size = 16)

# Actuals vs Predictions for SARIMA(1, 0, 1)(1, 0, 1, 5)
ax[1].plot(test_df["spx_ret"], color = "blue", label = "Actuals")
ax[1].plot(seas_pred, color = "red", label = "SARIMA(1, 0, 1)(1, 0, 1, 5) Predictions")
ax[1].set_title(f"SARIMA(1, 0, 1)(1, 0, 1, 5) Predictions (RMSE: {np.round(sarima_rmse, 3)})", size = 16)

# Displaying the plots
plt.show()

# FORECASTING AND CONFIDENCE INTERVALS
# Generating Forecast object
forecasts = model_results.get_forecast(len(test_df.index))
# Generating confidence intervals
forecasts_df = forecasts.conf_int(alpha = 0.05)  # Confidence Interval of 95%
# Actual predictions
forecasts_df["Predictions"] = model_results.predict(start = test_df.index[0], end = test_df.index[-1])

# Displaying first 5 rows of the forecasts_df
print(forecasts_df.head())

# RMSE of the forecasts
arima_rmse = np.sqrt(mean_squared_error(y_true = test_df["spx_ret"].values, y_pred = forecasts_df["Predictions"].values))


# PLOTTING THE FORECASTS AND CONFIDENCE INTERVALS
# Setting the figure size
plt.rcParams["figure.figsize"] = 18, 5

# Actual values of the S&P 500 returns in the test set
plt.plot(test_df["spx_ret"], color = "blue", label = "Actual Values")

# Predictions from the model and confidence intervals
plt.plot(forecasts_df["Predictions"], color = "red", label = "Predictions")
plt.plot(forecasts_df["upper spx_ret"], color = "green", linestyle = "--", label = "Conidence Levels(95%)")
plt.plot(forecasts_df["lower spx_ret"], color = "green", linestyle = "--")

# Title of the plot
plt.title(f"Predictions vs Actuals for ARIMA(1, 0, 1) Model (RMSE - {round(arima_rmse, 2)})", size = 24)

# Display the labels
plt.legend()
# Display the plot
plt.show()

print("Training Set Shape - ", train_df.shape)
print("Testing Set Shape - ", test_df.shape)

# Importing Required Package
import statsmodels.graphics.tsaplots as sgt

# Setting the figure size
plt.rcParams["figure.figsize"] = 12, 5

# PCF Plot for S&P 500 Returns
sgt.plot_pacf(train_df.spx_ret[1:], zero = False, lags = 40)

# Displaying the plot
plt.show()


print("Training Set Shape - ", train_df.shape)
print("Testing Set Shape - ", test_df.shape)

# Importing Required Package
import statsmodels.graphics.tsaplots as sgt

# Fixing plot size
plt.rcParams["figure.figsize"] = 18, 5

# Defining Subplots
fig, axes = plt.subplots(1, 2)

# Plotting ACF and PACF for S&P 500 Returns
sgt.plot_acf(train_df.spx_ret[1:], zero = False, lags = 40, ax = axes[0])
sgt.plot_pacf(train_df.spx_ret[1:], zero = False, lags = 40, ax = axes[1])

# Display the Plot
plt.show()

# MODEL FITTING
# Importing Required Package
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Defining the Model
model = SARIMAX(train_df["spx_ret"][1:], order = (1, 0, 1))
# Fitting the Model
model_results = model.fit()

# Printing the model summary
print(model_results.summary())

# FORECASTING
# Building Forecast Object to generate Confidence Intervals
arma_forecast = model_results.get_forecast(len(test_df.index))
arma_predictions_df = arma_forecast.conf_int(alpha = 0.05) # Confidence level of 95%
# Predictions
arma_predictions_df["Predictions"] = model_results.predict(start = test_df.index[0], end = test_df.index[-1])

# RMSE for the Predictions
arma_rmse = np.sqrt(mean_squared_error(test_df["spx_ret"].values, arma_predictions_df["Predictions"]))


# PLOTTING FORECASTS

# Set the Size of the figure
plt.rcParams["figure.figsize"] = 18, 5

# Plot the Actuals
plt.plot(test_df["spx_ret"], color = "blue", label = "Actual Values")

# Plot the Forecasts and the Confidence Intervals 
plt.plot(arma_predictions_df["Predictions"][test_df.index], color = "red", label = "Predictions")
plt.plot(arma_predictions_df["upper spx_ret"][test_df.index], color = "green", linestyle = "--", label = "Conidence Levels(95%)")
plt.plot(arma_predictions_df["lower spx_ret"][test_df.index], color = "green", linestyle = "--")

# Set the Title of the Plot
plt.title(f"Predictions(95% confidence) vs Actuals for ARMA(1, 1) Model (MSE - {round(arma_rmse, 2)})", size = 24)

# Display the plot with appropriate labels
plt.legend()
plt.show()

# Set the figure size
plt.rcParams["figure.figsize"] = 18, 5

# Plotting residuals
plt.plot(model_results.resid, label = "Residuals")

# Setting Title
plt.title("Residuals of ARMA(1, 1)", size = 24)

# Display the plot
plt.show()

print("Training Set Shape - ", train_df.shape)
print("Testing Set Shape - ", test_df.shape)

# SETTING THE PLOT SIZE AND SUBPLOTS
plt.rcParams["figure.figsize"] = 18, 3
fig, axes = plt.subplots(1, 3)


# ACF AND PACF PLOT FOR THE TRANSFORMED DATA
sgt.plot_acf(train_df["spx"].dropna(), zero = False, lags = 40, ax = axes[0])
axes[0].set_title("ACF", size = 24)
sgt.plot_pacf(train_df["spx"].dropna(), zero = False, lags = 40, ax = axes[1])
axes[1].set_title("PACF", size = 24)


# LINE PLOT OF THE TRANSFORMED DATA
axes[2].plot(train_df["spx"].dropna())
axes[2].set_title("S&P 500 Prices", size = 24)
plt.show()