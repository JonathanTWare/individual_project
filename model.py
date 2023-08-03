import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def holt_linear_model_fit(assault_train_df, target_col='ASSAULT', predictor_lag25_col='ASSAULT_Lag_25', predictor_lag80_col='ASSAULT_Lag_80', test_size=0.2):
    target = assault_train_df[target_col].shift(-365)
    predictor_lag25 = assault_train_df[predictor_lag25_col]
    predictor_lag80 = assault_train_df[predictor_lag80_col]

    predictors = pd.concat([predictor_lag25, predictor_lag80], axis=1)

    target = target.dropna()
    predictors = predictors.dropna()

    min_samples = min(len(target), len(predictors))
    target = target[:min_samples]
    predictors = predictors[:min_samples]

    target = target.dropna()

    train_predictors, test_predictors, train_target, test_target = train_test_split(predictors, target, test_size=test_size, random_state=123)

    holt_linear_model = sm.tsa.Holt(train_target, damped=True)
    holt_linear_fit = holt_linear_model.fit()

    predictions = holt_linear_fit.forecast(len(test_target))

    mse = mean_squared_error(test_target, predictions) ** 0.5

    metrics_test = pd.DataFrame({
        'Model': ['Holts Linear TEST'],
        'Target': ['ASSAULT'],
        'RMSE': [mse]
    })

    return metrics_test


def holt_linear_model_fit_battery(battery_train_df, target_col='BATTERY', predictor_lag83_col='BATTERY_Lag_83', test_size=0.2):
    target = battery_train_df[target_col].shift(-365)
    predictor_lag83 = battery_train_df[predictor_lag83_col]

    predictors = pd.concat([predictor_lag83], axis=1)

    target = target.dropna()
    predictors = predictors.dropna()

    min_samples = min(len(target), len(predictors))
    target = target[:min_samples]
    predictors = predictors[:min_samples]

    target = target.dropna()

    train_predictors, test_predictors, train_target, test_target = train_test_split(predictors, target, test_size=test_size, random_state=123)

    holt_linear_model = sm.tsa.Holt(train_target, damped=True)
    holt_linear_fit = holt_linear_model.fit()

    predictions = holt_linear_fit.forecast(len(test_target))

    mse = mean_squared_error(test_target, predictions) ** 0.5

    metrics_test = pd.DataFrame({
        'Model': ['Holts Linear TEST'],
        'Target': ['BATTERY'],
        'RMSE': [mse]
    })

    return metrics_test

def simple_exponential_model_fit(criminal_damage_train_df, target_col='CRIMINAL DAMAGE', predictor_lag25_col='CRIMINAL_DAMAGE_Lag_25', predictor_lag45_col='CRIMINAL_DAMAGE_Lag_45', predictor_lag75_col='CRIMINAL_DAMAGE_Lag_75', test_size=0.2):
    target = criminal_damage_train_df[target_col].shift(-365)
    predictor_lag25 = criminal_damage_train_df[predictor_lag25_col]
    predictor_lag45 = criminal_damage_train_df[predictor_lag45_col]
    predictor_lag75 = criminal_damage_train_df[predictor_lag75_col]

    predictors = pd.concat([predictor_lag25, predictor_lag45, predictor_lag75], axis=1)

    target = target.dropna()
    predictors = predictors.dropna()

    min_samples = min(len(target), len(predictors))
    target = target[:min_samples]
    predictors = predictors[:min_samples]

    target = target.dropna()

    train_predictors, test_predictors, train_target, test_target = train_test_split(predictors, target, test_size=test_size, random_state=123)

    simple_exponential_model = sm.tsa.SimpleExpSmoothing(train_target)
    simple_exponential_fit = simple_exponential_model.fit()

    predictions = simple_exponential_fit.forecast(len(test_target))

    mse = mean_squared_error(test_target, predictions) ** 0.5

    metrics_test = pd.DataFrame({
        'Model': ['Simple Exponential Smoothing TEST'],
        'Target': ['CRIMINAL DAMAGE'],
        'RMSE': [mse]
    })
    
    return metrics_test


def plot_data_over_time(train):
    train['Date'] = pd.to_datetime(train.index)
    train.set_index('Date', inplace=True)

    data_weekly = train.resample('W').sum()
    data_monthly = train.resample('M').sum()
    data_yearly = train.resample('Y').sum()

    columns = train.columns

    for column in columns:
        plt.figure(figsize=(10, 5))
        plt.plot(train.index, train[column], label='Daily', linestyle='-', marker='o')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.title(f'{column} over Time - Daily')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(data_weekly.index, data_weekly[column], label='Weekly', linestyle='--', marker='o')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.title(f'{column} over Time - Weekly')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(data_monthly.index, data_monthly[column], label='Monthly', linestyle='-.', marker='o')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.title(f'{column} over Time - Monthly')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(data_yearly.index, data_yearly[column], label='Yearly', linestyle=':', marker='o')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.title(f'{column} over Time - Yearly')
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_autocorrelation(train, crime_types):
    # Resample the data to daily frequency and calculate the mean for each crime type
    train_resampled = train[crime_types].resample('W').mean()

    # Loop through the crime types and create autocorrelation plots
    for crime_type in crime_types:
        plt.figure(figsize=(10, 5))
        pd.plotting.autocorrelation_plot(train_resampled[crime_type])
        plt.xlabel("Lags")
        plt.ylabel("Autocorrelation")
        plt.title(f"Autocorrelation Plot for {crime_type}")
        plt.show()
        
def create_lagged_features(train, crime_types, max_lag=365):
    for crime_type in crime_types:
        lags = range(1, max_lag + 1)
        for lag in lags:
            train[f'{crime_type}_Lag_{lag}'] = train[crime_type].shift(lag)
    return train


import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def evaluate_assault_model(assault_train_df, target_col='ASSAULT', predictor_lag25_col='ASSAULT_Lag_25', predictor_lag80_col='ASSAULT_Lag_80'):
    target = assault_train_df[target_col].shift(-365)
    predictor_lag25 = assault_train_df[predictor_lag25_col]
    predictor_lag80 = assault_train_df[predictor_lag80_col]

    predictors = pd.concat([predictor_lag25, predictor_lag80], axis=1)

    target = target.dropna()
    predictors = predictors.dropna()

    min_samples = min(len(target), len(predictors))
    target = target[:min_samples]
    predictors = predictors[:min_samples]

    target = target.dropna()

    train_ratio = 0.8
    train_size = int(train_ratio * len(target))

    train_target = target[:train_size]
    train_predictors = predictors[:train_size]
    test_target = target[train_size:]
    test_predictors = predictors[train_size:]

    model = LinearRegression()
    model.fit(train_predictors, train_target)

    predictions = model.predict(test_predictors)

    mse = mean_squared_error(test_target, predictions) ** 0.5

    metrics_assault = pd.DataFrame({
        'Model': ['Last Observed Baseline'],
        'Target': 'ASSAULT',
        'RMSE': [mse]
    })

    historical_average = target.mean()

    predictions_simple_avg = [historical_average] * len(test_target)

    mse_simple_avg = mean_squared_error(test_target, predictions_simple_avg) ** 0.5

    metrics_assault = metrics_assault.append({
        'Model': 'Simple Average Baseline',
        'Target': 'ASSAULT',
        'RMSE': mse_simple_avg
    }, ignore_index=True)

    window_size = 30
    moving_average = target.rolling(window=window_size).mean().dropna()

    last_value = moving_average.iloc[-1]
    predictions_moving_avg = [last_value] * len(test_target)

    mse_moving_avg = mean_squared_error(test_target, predictions_moving_avg) ** 0.5

    metrics_assault = metrics_assault.append({
        'Model': 'Moving Average Baseline (Window Size {})'.format(window_size),
        'Target': 'ASSAULT',
        'RMSE': mse_moving_avg
    }, ignore_index=True)

    simple_exponential_model = sm.tsa.SimpleExpSmoothing(train_target)
    simple_exponential_fit = simple_exponential_model.fit()

    predictions_simple_exp = simple_exponential_fit.forecast(len(test_target))

    mse_simple_exp = mean_squared_error(test_target, predictions_simple_exp) ** 0.5

    metrics_assault = metrics_assault.append({
        'Model': 'Simple Exponential Smoothing',
        'Target': 'ASSAULT',
        'RMSE': mse_simple_exp
    }, ignore_index=True)

    holt_linear_model = sm.tsa.Holt(train_target, damped=True)
    holt_linear_fit = holt_linear_model.fit()

    predictions_holt_linear = holt_linear_fit.forecast(len(test_target))

    mse_holt_linear = mean_squared_error(test_target, predictions_holt_linear) ** 0.5

    metrics_assault = metrics_assault.append({
        'Model': 'Holts Linear Trend Forecasting',
        'Target': 'ASSAULT',
        'RMSE': mse_holt_linear
    }, ignore_index=True)

    return metrics_assault


def evaluate_battery_model(battery_train_df, target_col='BATTERY', predictor_lag83_col='BATTERY_Lag_83'):
    target = battery_train_df[target_col].shift(-365)
    predictor_lag83 = battery_train_df[predictor_lag83_col]
    

    predictors = pd.concat([predictor_lag83], axis=1)

    target = target.dropna()
    predictors = predictors.dropna()

    min_samples = min(len(target), len(predictors))
    target = target[:min_samples]
    predictors = predictors[:min_samples]

    target = target.dropna()

    train_ratio = 0.8
    train_size = int(train_ratio * len(target))

    train_target = target[:train_size]
    train_predictors = predictors[:train_size]
    test_target = target[train_size:]
    test_predictors = predictors[train_size:]

    model = LinearRegression()
    model.fit(train_predictors, train_target)

    predictions = model.predict(test_predictors)

    mse = mean_squared_error(test_target, predictions) ** 0.5

    metrics_battery = pd.DataFrame({
        'Model': ['Last Observed Baseline'],
        'Target': 'BATTERY',
        'RMSE': [mse]
    })

    historical_average = target.mean()

    predictions_simple_avg = [historical_average] * len(test_target)

    mse_simple_avg = mean_squared_error(test_target, predictions_simple_avg) ** 0.5

    metrics_battery = metrics_battery.append({
        'Model': 'Simple Average Baseline',
        'Target': 'BATTERY',
        'RMSE': mse_simple_avg
    }, ignore_index=True)

    window_size = 30
    moving_average = target.rolling(window=window_size).mean().dropna()

    last_value = moving_average.iloc[-1]
    predictions_moving_avg = [last_value] * len(test_target)

    mse_moving_avg = mean_squared_error(test_target, predictions_moving_avg) ** 0.5

    metrics_battery = metrics_battery.append({
        'Model': 'Moving Average Baseline (Window Size {})'.format(window_size),
        'Target': 'BATTERY',
        'RMSE': mse_moving_avg
    }, ignore_index=True)

    simple_exponential_model = sm.tsa.SimpleExpSmoothing(train_target)
    simple_exponential_fit = simple_exponential_model.fit()

    predictions_simple_exp = simple_exponential_fit.forecast(len(test_target))

    mse_simple_exp = mean_squared_error(test_target, predictions_simple_exp) ** 0.5

    metrics_battery = metrics_battery.append({
        'Model': 'Simple Exponential Smoothing',
        'Target': 'BATTERY',
        'RMSE': mse_simple_exp
    }, ignore_index=True)

    holt_linear_model = sm.tsa.Holt(train_target, damped=True)
    holt_linear_fit = holt_linear_model.fit()

    predictions_holt_linear = holt_linear_fit.forecast(len(test_target))

    mse_holt_linear = mean_squared_error(test_target, predictions_holt_linear) ** 0.5

    metrics_battery = metrics_battery.append({
        'Model': 'Holts Linear Trend Forecasting',
        'Target': 'BATTERY',
        'RMSE': mse_holt_linear
    }, ignore_index=True)

    return metrics_battery


def evaluate_criminaldamage_model(criminal_damage_train_df, target_col='CRIMINAL DAMAGE', predictor_lag25_col='CRIMINAL_DAMAGE_Lag_25', predictor_lag45_col='CRIMINAL_DAMAGE_Lag_45', predictor_lag75_col='CRIMINAL_DAMAGE_Lag_75'):
    target = criminal_damage_train_df[target_col].shift(-365)
    predictor_lag25 = criminal_damage_train_df[predictor_lag25_col]
    predictor_lag45 = criminal_damage_train_df[predictor_lag45_col]
    predictor_lag75 = criminal_damage_train_df[predictor_lag75_col]

    predictors = pd.concat([predictor_lag25, predictor_lag45, predictor_lag75], axis=1)

    target = target.dropna()
    predictors = predictors.dropna()

    min_samples = min(len(target), len(predictors))
    target = target[:min_samples]
    predictors = predictors[:min_samples]

    target = target.dropna()

    train_ratio = 0.8
    train_size = int(train_ratio * len(target))

    train_target = target[:train_size]
    train_predictors = predictors[:train_size]
    test_target = target[train_size:]
    test_predictors = predictors[train_size:]

    model = LinearRegression()
    model.fit(train_predictors, train_target)

    predictions = model.predict(test_predictors)

    mse = mean_squared_error(test_target, predictions) ** 0.5

    metrics_criminal = pd.DataFrame({
        'Model': ['Last Observed Baseline'],
        'Target': 'CRIMINAL DAMAGE',
        'RMSE': [mse]
    })

    historical_average = target.mean()

    predictions_simple_avg = [historical_average] * len(test_target)

    mse_simple_avg = mean_squared_error(test_target, predictions_simple_avg) ** 0.5

    metrics_criminal = metrics_criminal.append({
        'Model': 'Simple Average Baseline',
        'Target': 'CRIMINAL DAMAGE',
        'RMSE': mse_simple_avg
    }, ignore_index=True)

    window_size = 30
    moving_average = target.rolling(window=window_size).mean().dropna()

    last_value = moving_average.iloc[-1]
    predictions_moving_avg = [last_value] * len(test_target)

    mse_moving_avg = mean_squared_error(test_target, predictions_moving_avg) ** 0.5

    metrics_criminal = metrics_criminal.append({
        'Model': 'Moving Average Baseline (Window Size {})'.format(window_size),
        'Target': 'CRIMINAL DAMAGE',
        'RMSE': mse_moving_avg
    }, ignore_index=True)

    simple_exponential_model = sm.tsa.SimpleExpSmoothing(train_target)
    simple_exponential_fit = simple_exponential_model.fit()

    predictions_simple_exp = simple_exponential_fit.forecast(len(test_target))

    mse_simple_exp = mean_squared_error(test_target, predictions_simple_exp) ** 0.5

    metrics_criminal = metrics_criminal.append({
        'Model': 'Simple Exponential Smoothing',
        'Target': 'CRIMINAL DAMAGE',
        'RMSE': mse_simple_exp
    }, ignore_index=True)

    holt_linear_model = sm.tsa.Holt(train_target, damped=True)
    holt_linear_fit = holt_linear_model.fit()

    predictions_holt_linear = holt_linear_fit.forecast(len(test_target))

    mse_holt_linear = mean_squared_error(test_target, predictions_holt_linear) ** 0.5

    metrics_criminal = metrics_criminal.append({
        'Model': 'Holts Linear Trend Forecasting',
        'Target': 'CRIMINAL DAMAGE',
        'RMSE': mse_holt_linear
    }, ignore_index=True)

    return  metrics_criminal

def create_criminal_damage_train_df(train):
    target_col = 'CRIMINAL DAMAGE'
    lagged_feature_cols = [f'CRIMINAL_DAMAGE_Lag_{lag}' for lag in range(1, 366)]
    columns_to_check = [target_col] + lagged_feature_cols

    missing_columns = set(columns_to_check) - set(train.columns)
    if missing_columns:
        raise KeyError(f"Columns {missing_columns} are missing in the train DataFrame.")

    criminal_damage_train_df = train[columns_to_check].copy()

    # Drop rows with NaN values in any of the columns
    criminal_damage_train_df.dropna(inplace=True)

    return criminal_damage_train_df