'''
Author: jyyd23@mails.tsinghua.edu.cn
Date: 2024-05-21 22:58:40
LastEditors: jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-25 00:46:54
FilePath: PM_DL.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(rc={'figure.dpi': 600}) # 设置dpi为300
import matplotlib.pyplot as plt
import dataframe_image as dfi
import pylab
import time

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict,cross_validate, RandomizedSearchCV
# from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler,  MinMaxScaler

import test_pnc_model.model_test as model_test
from scikeras.wrappers import KerasRegressor
from keras.optimizers import Adam
from tensorflow import keras

import PM_stack as stack

def pred_plot(y_test, y_pred, resid):
    print('----------------------------------')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print('MSE: ', mse)
    print('MAE: ', mae)
    print('r2 score: ', r2)
    print('Explained_variance: ', evs)
    return mse, mae, r2, evs

def train_RNN_PM25():
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stack.stander_data(workstation_Flag)
    
    x_train, y_train, x_test, y_test = pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test

    epochs_values = [100]
    batch_sizes = [128, 64, 32]
    verbose_value = 2
    random_state_value = 42
    val_sizes = [0.1]

    results = []

    if not os.path.exists('./test_deepmodel/pm25/'):
        os.makedirs('./test_deepmodel/pm25/')
    if not os.path.exists('./test_deepmodel/pm25/figure/'):
        os.makedirs('./test_deepmodel/pm25/figure/')
    if not os.path.exists('./test_deepmodel/pm25/model/'):
        os.makedirs('./test_deepmodel/pm25/model/')
    if not os.path.exists('./test_deepmodel/pm25/tab/'):
        os.makedirs('./test_deepmodel/pm25/tab/')
    

    for val_size in val_sizes:
        x_train_mt, y_train_mt, x_val_mt, y_val_mt, x_test_mt, y_test_mt = model_test.val_in(x_train, x_test,
                                                                                            y_train, y_test,
                                                                                            val_size=val_size)

        for epoch in epochs_values:
            for batch in batch_sizes:
                rnn_regr = KerasRegressor(build_fn=model_test.build_rnn_model(input_shape=(1, 13)), 
                                        epochs=epoch, batch_size=batch, verbose=verbose_value, random_state=random_state_value)
                rnn_fit = rnn_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
                
                rnn_pred = rnn_fit.predict(x_test_mt)
                rnn_resid = rnn_pred - y_test_mt
                
                mse, mae, r2, evs = pred_plot(y_test_mt, rnn_pred, rnn_resid)

                plt.figure(figsize=(10, 5), dpi=600)
                plt.plot(rnn_fit.history_['loss'], label='train')
                plt.plot(rnn_fit.history_['val_loss'], label='val')
                plt.ylabel('RNN Loss')
                plt.xlabel('RNN Epochs')
                plt.title(f'RNN Training and Validation Loss (val_size={val_size}, epochs={epoch}, batch={batch})')
                plt.legend()

                plot_name = f"./test_deepmodel/pm25/figure/rnn_loss_valsize_{val_size}_epochs_{epoch}_batch_{batch}.png"
                plt.savefig(plot_name)
                plt.close()

                results.append({
                    'val_size': val_size,
                    'epochs': epoch,
                    'batch_size': batch,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'evs': evs
                })
    df_results = pd.DataFrame(results, index=None, columns=['val_size', 'epochs', 'batch_size', 'mse', 'mae', 'r2', 'evs'])
    csv_name = "./test_deepmodel/pm25/tab/results_rnn_model_valsize_{val_size}_epochs_{epoch}_batch_{batch}.csv"
    df_results.to_csv(csv_name, index=False)


def train_RNN_PM10():
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stack.stander_data(workstation_Flag)
    
    x_train, y_train, x_test, y_test = pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test

    epochs_values = [100]
    batch_sizes = [128, 64, 32]
    verbose_value = 2
    random_state_value = 42
    val_sizes = [0.1]

    results = []

    for val_size in val_sizes:
        x_train_mt, y_train_mt, x_val_mt, y_val_mt, x_test_mt, y_test_mt = model_test.val_in(x_train, x_test,
                                                                                            y_train, y_test,
                                                                                            val_size=val_size)

        for epoch in epochs_values:
            for batch in batch_sizes:
                rnn_regr = KerasRegressor(build_fn=model_test.build_rnn_model(input_shape=(1, 13)), 
                                        epochs=epoch, batch_size=batch, verbose=verbose_value, random_state=random_state_value)
                rnn_fit = rnn_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
                
                rnn_pred = rnn_fit.predict(x_test_mt)
                rnn_resid = rnn_pred - y_test_mt
                
                mse, mae, r2, evs = pred_plot(y_test_mt, rnn_pred, rnn_resid)

                plt.figure(figsize=(10, 5), dpi=600)
                plt.plot(rnn_fit.history_['loss'], label='train')
                plt.plot(rnn_fit.history_['val_loss'], label='val')
                plt.ylabel('RNN Loss')
                plt.xlabel('RNN Epochs')
                plt.title(f'RNN Training and Validation Loss (val_size={val_size}, epochs={epoch}, batch={batch})')
                plt.legend()

                plot_name = f"./test_deepmodel/pm10/figure/rnn_loss_valsize_{val_size}_epochs_{epoch}_batch_{batch}.png"
                plt.savefig(plot_name)
                plt.close()

                results.append({
                    'val_size': val_size,
                    'epochs': epoch,
                    'batch_size': batch,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'evs': evs
                })
    df_results = pd.DataFrame(results, index=None, columns=['val_size', 'epochs', 'batch_size', 'mse', 'mae', 'r2', 'evs'])
    csv_name = "./test_deepmodel/pm10/tab/results_rnn_model_valsize_{val_size}_epochs_{epoch}_batch_{batch}.csv"
    df_results.to_csv(csv_name, index=False)


def train_CNN_PM25():
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stack.stander_data(workstation_Flag)
    
    x_train, y_train, x_test, y_test = pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test

    epochs_values = [100]
    batch_sizes = [128, 64, 32]
    verbose_value = 2
    random_state_value = 42
    val_sizes = [0.1]

    results = []

    for val_size in val_sizes:
        x_train_mt, y_train_mt, x_val_mt, y_val_mt, x_test_mt, y_test_mt = model_test.val_in(x_train, x_test,
                                                                                            y_train, y_test,
                                                                                            val_size=val_size)

        for epoch in epochs_values:
            for batch in batch_sizes:
                cnn_regr = KerasRegressor(build_fn=model_test.build_cnn_model(input_shape=(1, 13)), 
                                        epochs=epoch, batch_size=batch, verbose=verbose_value, random_state=random_state_value)
                cnn_fit = cnn_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
                
                cnn_pred = cnn_fit.predict(x_test_mt)
                cnn_resid = cnn_pred - y_test_mt
                
                mse, mae, r2, evs = pred_plot(y_test_mt, cnn_pred, cnn_resid)

                plt.figure(figsize=(10, 5), dpi=600)
                plt.plot(cnn_fit.history_['loss'], label='train')
                plt.plot(cnn_fit.history_['val_loss'], label='val')
                plt.ylabel('CNN Loss')
                plt.xlabel('CNN Epochs')
                plt.title(f'CNN Training and Validation Loss (val_size={val_size}, epochs={epoch}, batch={batch})')
                plt.legend()

                plot_name = f"./test_deepmodel/pm25/figure/cnn_loss_valsize_{val_size}_epochs_{epoch}_batch_{batch}.png"
                plt.savefig(plot_name)
                plt.close()

                results.append({
                    'val_size': val_size,
                    'epochs': epoch,
                    'batch_size': batch,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'evs': evs
                })
    df_results = pd.DataFrame(results, index=None, columns=['val_size', 'epochs', 'batch_size', 'mse', 'mae', 'r2', 'evs'])
    csv_name = "./test_deepmodel/pm25/tab/results_cnn_model_valsize_{val_size}_epochs_{epoch}_batch_{batch}.csv"
    df_results.to_csv(csv_name, index=False)

def train_CNN_PM10():
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stack.stander_data(workstation_Flag)
    
    x_train, y_train, x_test, y_test = pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test

    epochs_values = [100]
    batch_sizes = [128, 64, 32]
    verbose_value = 2
    random_state_value = 42
    val_sizes = [0.1]

    results = []

    for val_size in val_sizes:
        x_train_mt, y_train_mt, x_val_mt, y_val_mt, x_test_mt, y_test_mt = model_test.val_in(x_train, x_test,
                                                                                            y_train, y_test,
                                                                                            val_size=val_size)

        for epoch in epochs_values:
            for batch in batch_sizes:
                cnn_regr = KerasRegressor(build_fn=model_test.build_cnn_model(input_shape=(1, 13)), 
                                        epochs=epoch, batch_size=batch, verbose=verbose_value, random_state=random_state_value)
                cnn_fit = cnn_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
                
                cnn_pred = cnn_fit.predict(x_test_mt)
                cnn_resid = cnn_pred - y_test_mt
                
                mse, mae, r2, evs = pred_plot(y_test_mt, cnn_pred, cnn_resid)

                plt.figure(figsize=(10, 5), dpi=600)
                plt.plot(cnn_fit.history_['loss'], label='train')
                plt.plot(cnn_fit.history_['val_loss'], label='val')
                plt.ylabel('CNN Loss')
                plt.xlabel('CNN Epochs')
                plt.title(f'CNN Training and Validation Loss (val_size={val_size}, epochs={epoch}, batch={batch})')
                plt.legend()

                plot_name = f"./test_deepmodel/pm10/figure/cnn_loss_valsize_{val_size}_epochs_{epoch}_batch_{batch}.png"
                plt.savefig(plot_name)
                plt.close()

                results.append({
                    'val_size': val_size,
                    'epochs': epoch,
                    'batch_size': batch,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'evs': evs
                })
    df_results = pd.DataFrame(results, index=None, columns=['val_size', 'epochs', 'batch_size', 'mse', 'mae', 'r2', 'evs'])
    csv_name = "./test_deepmodel/pm10/tab/results_cnn_model_valsize_{val_size}_epochs_{epoch}_batch_{batch}.csv"
    df_results.to_csv(csv_name, index=False)

def train_LSTM_PM25():
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stack.stander_data(workstation_Flag)
    
    x_train, y_train, x_test, y_test = pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test

    epochs_values = [100]
    batch_sizes = [128, 64, 32]
    verbose_value = 2
    random_state_value = 42
    val_sizes = [0.1]

    results = []

    for val_size in val_sizes:
        x_train_mt, y_train_mt, x_val_mt, y_val_mt, x_test_mt, y_test_mt = model_test.val_in(x_train, x_test,
                                                                                            y_train, y_test,
                                                                                            val_size=val_size)

        for epoch in epochs_values:
            for batch in batch_sizes:
                lstm_regr = KerasRegressor(build_fn=model_test.build_lstm_model(input_shape=(1, 13)), 
                                        epochs=epoch, batch_size=batch, verbose=verbose_value, random_state=random_state_value)
                lstm_fit = lstm_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
                
                lstm_pred = lstm_fit.predict(x_test_mt)
                lstm_resid = lstm_pred - y_test_mt
                
                mse, mae, r2, evs = pred_plot(y_test_mt, lstm_pred, lstm_resid)

                plt.figure(figsize=(10, 5), dpi=600)
                plt.plot(lstm_fit.history_['loss'], label='train')
                plt.plot(lstm_fit.history_['val_loss'], label='val')
                plt.ylabel('LSTM Loss')
                plt.xlabel('LSTM Epochs')
                plt.title(f'LSTM Training and Validation Loss (val_size={val_size}, epochs={epoch}, batch={batch})')
                plt.legend()

                plot_name = f"./test_deepmodel/pm25/figure/lstm_loss_valsize_{val_size}_epochs_{epoch}_batch_{batch}.png"
                plt.savefig(plot_name)
                plt.close()

                results.append({
                    'val_size': val_size,
                    'epochs': epoch,
                    'batch_size': batch,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'evs': evs
                })

    df_results = pd.DataFrame(results, index=None, columns=['val_size', 'epochs', 'batch_size', 'mse', 'mae', 'r2', 'evs'])
    csv_name = "./test_deepmodel/pm25/tab/results_lstm_model_valsize_{val_size}_epochs_{epoch}_batch_{batch}.csv"
    df_results.to_csv(csv_name, index=False)

def train_LSTM_PM10():
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stack.stander_data(workstation_Flag)
    
    x_train, y_train, x_test, y_test = pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test

    epochs_values = [100]
    batch_sizes = [128, 64, 32]
    verbose_value = 2
    random_state_value = 42
    val_sizes = [0.1]

    results = []

    for val_size in val_sizes:
        x_train_mt, y_train_mt, x_val_mt, y_val_mt, x_test_mt, y_test_mt = model_test.val_in(x_train, x_test,
                                                                                            y_train, y_test,
                                                                                            val_size=val_size)

        for epoch in epochs_values:
            for batch in batch_sizes:
                lstm_regr = KerasRegressor(build_fn=model_test.build_lstm_model(input_shape=(1, 13)), 
                                        epochs=epoch, batch_size=batch, verbose=verbose_value, random_state=random_state_value)
                lstm_fit = lstm_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
                
                lstm_pred = lstm_fit.predict(x_test_mt)
                lstm_resid = lstm_pred - y_test_mt
                
                mse, mae, r2, evs = pred_plot(y_test_mt, lstm_pred, lstm_resid)

                plt.figure(figsize=(10, 5), dpi=600)
                plt.plot(lstm_fit.history_['loss'], label='train')
                plt.plot(lstm_fit.history_['val_loss'], label='val')
                plt.ylabel('LSTM Loss')
                plt.xlabel('LSTM Epochs')
                plt.title(f'LSTM Training and Validation Loss (val_size={val_size}, epochs={epoch}, batch={batch})')
                plt.legend()

                plot_name = f"./test_deepmodel/pm10/figure/lstm_loss_valsize_{val_size}_epochs_{epoch}_batch_{batch}.png"
                plt.savefig(plot_name)
                plt.close()

                results.append({
                    'val_size': val_size,
                    'epochs': epoch,
                    'batch_size': batch,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'evs': evs
                })

    df_results = pd.DataFrame(results, index=None, columns=['val_size', 'epochs', 'batch_size', 'mse', 'mae', 'r2', 'evs'])
    csv_name = "./test_deepmodel/pm10/tab/results_lstm_model_valsize_{val_size}_epochs_{epoch}_batch_{batch}.csv"
    df_results.to_csv(csv_name, index=False)

def pred_PM25():
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stack.stander_data(workstation_Flag)
    pm_feature_path = '../../code/dataset/NABEL/feature_data/'
    pmdata2016_2019 = pd.read_csv(pm_feature_path + 'feature_data_2016_2019_PM.csv')
    pmdata2020 = pd.read_csv(pm_feature_path + 'feature_data_2020_PM.csv')
    pmdata2016_2019 = pmdata2016_2019.iloc[:,1:]
    pmdata2020 = pmdata2020.iloc[:,1:]
    ## PM2.5
    pmdata2016_2019_25 = pmdata2016_2019[['Date/time', 'station', 'PM2.5 [ug/m3]', 'PM10 [ug/m3]',
                                    'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                                    'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                                    'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                                    'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday']]
    pmdata2020_25 = pmdata2020[['Date/time', 'station', 'PM2.5 [ug/m3]', 'PM10 [ug/m3]',
                            'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                            'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                            'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                            'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday']]
    ## PM10
    pmdata2016_2019_10 = pmdata2016_2019[['Date/time', 'station', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]',
                                    'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                                    'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                                    'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                                    'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday']]
    pmdata2020_10 = pmdata2020[['Date/time', 'station', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]',
                            'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                            'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                            'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                            'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday']]

    val_size = 0.1
    x_train, y_train, x_test, y_test = pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test
    x_train_mt, y_train_mt, x_val_mt, y_val_mt, x_test_mt, y_test_mt = model_test.val_in(x_train, x_test,
                                                                                        y_train, y_test,
                                                                                        val_size=val_size)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    rnn_regr = KerasRegressor(build_fn=model_test.build_rnn_model(input_shape=(1, 13)), 
                                epochs=100, batch_size=128, verbose=2, random_state=42)
    rnn_fit = rnn_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
    data2020_pred1 = stack.pred_metrics(rnn_fit, x_test_mt, y_test_mt, pmdata2020_25)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    cnn_regr = KerasRegressor(build_fn=model_test.build_cnn_model(input_shape=(1, 13)), 
                                        epochs=100, batch_size=128, verbose=2, random_state=42)
    cnn_fit = cnn_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
    data2020_pred2 = stack.pred_metrics(cnn_fit, x_test_mt, y_test_mt, data2020_pred1)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    lstm_regr = KerasRegressor(build_fn=model_test.build_lstm_model(input_shape=(1, 13)), 
                                        epochs=100, batch_size=64, verbose=2, random_state=42)
    lstm_fit = lstm_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
    data2020_pred3 = stack.pred_metrics(lstm_fit , x_test_mt, y_test_mt, data2020_pred2)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    x_train_mt2, y_train_mt2, x_val_mt2, y_val_mt2, x_test_mt2, y_test_mt2 = model_test.val_in_trans(x_train, x_test,
                                                                                            y_train, y_test,
                                                                                            val_size=val_size)
    transform_regr = KerasRegressor(build_fn=model_test.build_transformer_model(input_shape=(1, 13)), 
                                        epochs=100, batch_size=128, verbose=2, random_state=42)
    transform_fit = transform_regr.fit(x_train_mt2, y_train_mt2, validation_data=(x_val_mt2, y_val_mt2))
    data2020_pred4 = stack.pred_metrics(transform_fit, x_test_mt2, y_test_mt2, data2020_pred3)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    data2020_pred4.columns = ['Date/time', 'station', 'PM2.5 [ug/m3]', 'PM10 [ug/m3]',
                            'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                            'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                            'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                            'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday',
                            'rnn_pred','rnn_resid','cnn_pred','cnn_resid',
                            'lstm_pred','lstm_resid','transform_pred','transform_resid']
    pred_table_path = './test_deepmodel/pm25/out/'
    data2020_pred4.to_csv(pred_table_path + 'pm25data_2020_pred_DL.csv', index=False)

def pred_PM10():
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stack.stander_data(workstation_Flag)
    val_size = 0.1
    x_train, y_train, x_test, y_test = pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test
    x_train_mt, y_train_mt, x_val_mt, y_val_mt, x_test_mt, y_test_mt = model_test.val_in(x_train, x_test,
                                                                                        y_train, y_test,
                                                                                        val_size=val_size)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    rnn_regr = KerasRegressor(build_fn=model_test.build_rnn_model(input_shape=(1, 13)), 
                                epochs=100, batch_size=128, verbose=2, random_state=42)
    rnn_fit = rnn_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
    data2020_pred1 = stack.pred_metrics(rnn_fit, x_test_mt, y_test_mt, pmdata2020_25)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    cnn_regr = KerasRegressor(build_fn=model_test.build_cnn_model(input_shape=(1, 13)), 
                                        epochs=100, batch_size=64, verbose=2, random_state=42)
    cnn_fit = cnn_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
    data2020_pred2 = stack.pred_metrics(cnn_fit, x_test_mt, y_test_mt, data2020_pred1)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    lstm_regr = KerasRegressor(build_fn=model_test.build_lstm_model(input_shape=(1, 13)), 
                                        epochs=100, batch_size=64, verbose=2, random_state=42)
    lstm_fit = lstm_regr.fit(x_train_mt, y_train_mt, validation_data=(x_val_mt, y_val_mt))
    data2020_pred3 = stack.pred_metrics(lstm_fit , x_test_mt, y_test_mt, data2020_pred2)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    x_train_mt2, y_train_mt2, x_val_mt2, y_val_mt2, x_test_mt2, y_test_mt2 = model_test.val_in_trans(x_train, x_test,
                                                                                            y_train, y_test,
                                                                                            val_size=val_size)
    transform_regr = KerasRegressor(build_fn=model_test.build_transformer_model(input_shape=(1, 13)), 
                                        epochs=100, batch_size=128, verbose=2, random_state=42)
    transform_fit = transform_regr.fit(x_train_mt2, y_train_mt2, validation_data=(x_val_mt2, y_val_mt2))
    data2020_pred4 = stack.pred_metrics(transform_fit, x_test_mt2, y_test_mt2, data2020_pred3)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    data2020_pred4.columns = ['Date/time', 'station', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]',
                            'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                            'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                            'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                            'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday',
                            'rnn_pred','rnn_resid','cnn_pred','cnn_resid',
                            'lstm_pred','lstm_resid','transform_pred','transform_resid']
    pred_table_path = './test_deepmodel/pm10/out/'
    data2020_pred4.to_csv(pred_table_path + 'pm10data_2020_pred_DL.csv', index=False)


