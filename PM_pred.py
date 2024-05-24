'''
Author: jyyd23@mails.tsinghua.edu.cn
Date: 2024-05-21 22:47:53
LastEditors: jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-25 00:48:31
FilePath: PM_pred.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import time
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score,cross_val_predict,cross_validate, RandomizedSearchCV
import pylab
import warnings
warnings.filterwarnings("ignore")
from joblib import dump, load
from tqdm import  trange
import netCDF4 as nc
import scipy.io as sio

import PM_stack as stack


def load_pnc_joblib(pm_model_name:str='Stacking', pred_var:str='PM2.5'):
    if pred_var == 'PM2.5':
        pm_regr = load('./model/stack_trainedModel_pm25.joblib')
    elif pred_var == 'PM10':
        pm_regr = load('./model/stack_trainedModel_pm10.joblib')
    return pm_regr

def cams_train_data(pollution_data, pollution:str='NOX'):
    pollution_data = np.reshape(pollution_data, (-1, 112041))
    pollution_data = pd.DataFrame(np.transpose(pollution_data))
    others = pollution_data.iloc[:, 1:]
    camsnp = np.array(pollution_data.iloc[:, 0]).reshape(112041, 1)
    return camsnp, others

def get_pnc_process_hour_data(args):
    hourId, pm_name = args
    workstation_Flag = False
    (pm_x_all_scaler_25, pm_x_all_scaler_10,
     _, _, _, _,
     _, _, _, _,
     _, _, _, _,) = stack.stander_data(workstation_Flag)
    pm_model_name='Stacking'
    pm25_regr = load_pnc_joblib(pm_model_name='Stacking',pred_var='PM2.5')
    pm10_regr = load_pnc_joblib(pm_model_name='Stacking',pred_var='PM10')
    noxdata = np.fromfile('../code/dataset/trainpredata/allbin/NOXtrainbin/' + str(hourId) + '_NOX_predData.bin', dtype=np.float32)
    no2data = np.fromfile('../code/dataset/trainpredata/allbin/NO2trainbin/' + str(hourId) + '_NO2_predData.bin', dtype=np.float32)
    pm10data = np.fromfile('../code/dataset/trainpredata/allbin/PM10trainbin/' + str(hourId) + '_PM10_predData.bin', dtype=np.float32)
    pm25data = np.fromfile('../code/dataset/trainpredata/allbin/PM2.5trainbin/' + str(hourId) + '_PM2.5_predData.bin', dtype=np.float32)
    o3data = np.fromfile('../code/dataset/trainpredata/allbin/O3trainbin/' + str(hourId) + '_O3_predData.bin', dtype=np.float32)
    camsnox, others = cams_train_data(noxdata, pollution='NOX')
    camsno2, _ = cams_train_data(no2data, pollution='NO2')
    camspm10, _ = cams_train_data(pm10data, pollution='PM10')
    camspm25, _ = cams_train_data(pm25data, pollution='PM2.5')
    camso3, _ = cams_train_data(o3data, pollution='O3')

    pred_pm = np.concatenate((camsnox, camsno2, camspm10, camspm25, camso3, others), axis=1)
    pred_pm = pd.DataFrame(pred_pm)
    pred_pm.columns = ['NOX [ug/m3 eq. NO2]', 'NO2 [ug/m3]', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]', 'O3 [ug/m3]', 'Radiation[W/m2]', 'Temperature',
                        'Precipitation[mm]', 'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour', 'month', 'weekday']
    pred_pm['NO2/NOX ratio'] = pred_pm['NO2 [ug/m3]'] / pred_pm['NOX [ug/m3 eq. NO2]']
    pred_pm25 = pred_pm[['PM10 [ug/m3]','NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio', 'O3 [ug/m3]', 'Radiation[W/m2]', 'Temperature',
                        'Precipitation[mm]', 'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour', 'month', 'weekday']]
    pred_pm10 = pred_pm[['PM2.5 [ug/m3]','NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio', 'O3 [ug/m3]', 'Radiation[W/m2]', 'Temperature',
                        'Precipitation[mm]', 'Relative humidity[%]', 'Wind speed[m/s]', 'trafficVol', 'hour', 'month', 'weekday']]
    pred_pm25.fillna(0, inplace=True)
    pred_pm10.fillna(0, inplace=True)
    pred_pm25 = np.array(pred_pm25)
    pred_pm10 = np.array(pred_pm10)
    x_pred_pm25 = pm_x_all_scaler_25.transform(pred_pm25)
    x_pred_pm10 = pm_x_all_scaler_10.transform(pred_pm10)
    pred_out_pm25 = pm25_regr.predict(x_pred_pm25)
    pred_out_pm10 = pm10_regr.predict(x_pred_pm10)
    pred_out_pm25 = np.array(  pred_out_pm25).reshape(112041, 1)
    pred_out_pm10 = np.array(pred_out_pm10).reshape(112041, 1)
    return  pred_out_pm25, pred_out_pm10


def get_matdata_pnc(hourId, pm_name):
    Delta_x = 0.01
    Delta_y = 0.01
    pm_model_name = 'Stacking'

    nf = nc.Dataset('../code/dataset/CAMS/CAMS_European_airquality_forecasts/SingleLevel_202101.nc')
    lon = np.array(nf.variables['longitude'][:]).reshape(-1, 1)
    lat = np.array(nf.variables['latitude'][:]).reshape(-1, 1)

    lonNew = np.arange(lon[0][0], lon[-1][0], Delta_x)
    latNew = np.arange(lat[0][0], lat[-1][0], -Delta_y)
    xnew, ynew = np.meshgrid(lonNew, latNew)
    # print(lonNew.shape)
    # print(latNew.shape)
    if pm_name == 'PM2.5':
        avgfilename = './out/pm25all/' + '2020_downScale_PM25_Stacking_' + str(hourId) + '.csv'
    elif pm_name == 'PM10':
        avgfilename = './out/pm10all/' + '2020_downScale_PM10_Stacking_' + str(hourId) + '.csv'
    else:
        print('ERROR')
    
    avgtemp = pd.read_csv(avgfilename, header=None)
    avgtemp = np.array(avgtemp).reshape(112041, 1)
    # temp= pd.read_csv(avgnoxfilename, header=None)
    value = np.array(avgtemp).reshape(531, 211).T


    avgConc = {
        'lonNew': np.array(lonNew),
        'latNew': np.array(latNew),
        'avgConc': np.array(value),
    }
    year = 2020
    if pm_name == 'PM2.5':
        filename = './out/matout/pm25/'+ str(year) + 'PM25_avgConc_Stacking_'+ str(hourId) + '.mat'
    elif pm_name == 'PM10':
        filename = './out/matout/pm10/'+ str(year) + 'PM10_avgConc_Stacking_' + str(hourId) + '.mat'
    sio.savemat(filename, avgConc)


def main():
    n = 8760
    year = 2020
    pm_name = 'PM2.5'
    for hourId in trange(100):
        args = hourId, pm_name
        pred_out_pm25, _ = get_pnc_process_hour_data(args)
        np.savetxt('./out/pm25all/2020_downScale_PM25_Stacking_' + str(hourId) + '.csv', pred_out_pm25, delimiter=',')
        get_matdata_pnc(hourId, pm_name)

    pm_name = 'PM10'
    for hourId in trange(100):
        args = hourId, pm_name
        _, pred_out_pm10 = get_pnc_process_hour_data(args)
        np.savetxt('./out/pm10all/2020_downScale_PM10_Stacking_' + str(hourId) + '.csv', pred_out_pm10, delimiter=',')
        get_matdata_pnc(hourId, pm_name)

if __name__ == '__main__':
    main()