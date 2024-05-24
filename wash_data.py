'''
Author: JYYD jyyd23@mails.tsinghua.edu.cn
Date: 2024-05-21 20:20:35
LastEditors: jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-25 00:46:31
FilePath: wash_data.py
'''
import os, sys
os.chdir(sys.path[0])
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd

def get_NABEL_data(workstation_Flag: bool = False):
    # Define the data path based on the workstation flag
    if workstation_Flag:
        NABEL_raw_data = '../code/dataset/NABEL/raw_data/'
    else:
        NABEL_raw_data = '../code/pncEstimator-main/data/NABEL/raw_data/'
    
    # List all files in the directory
    filelist = os.listdir(NABEL_raw_data)
    print('The number of NABEL stations: ', len(filelist))
    NABELdata = []
    for file in filelist:
        if file.startswith(('CHA', 'DAV')):
            continue
        filename = os.path.join(NABEL_raw_data, file)
        if filename.endswith('.csv'):
            data = pd.read_csv(filename, skiprows=6, sep=';')
            data_pm = data[['Date/time', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]',
                            'PM2.5 [ug/m3]', 'NOX [ug/m3 eq. NO2]', 'TEMP [C]', 'PREC [mm]',
                            'RAD [W/m2]']]
            data_pm['NO2/NOX ratio'] = data_pm['NO2 [ug/m3]'] / data_pm['NOX [ug/m3 eq. NO2]']
            data_pm['PM2.5/PM10 ratio'] = data_pm['PM2.5 [ug/m3]'] / data_pm['PM10 [ug/m3]']
            station = [file[0:3]] * len(data_pm)
            data_pm['station'] = station
            # Reorder columns
            data_pm = data_pm[['Date/time', 'station', 'NOX [ug/m3 eq. NO2]', 'NO2 [ug/m3]', 'NO2/NOX ratio',
                               'PM2.5/PM10 ratio',
                               'PM10 [ug/m3]', 'PM2.5 [ug/m3]', 'O3 [ug/m3]', 'TEMP [C]', 'PREC [mm]',
                               'RAD [W/m2]']]
            NABELdata.append(data_pm)
    NABELdata = pd.concat(NABELdata, axis=0).reset_index(drop=True)
    NABELdata['Date/time'] = pd.to_datetime(NABELdata['Date/time'], format='%d.%m.%Y %H:%M')
    print('++++++++++++++++++NABELdata++++++++++++++++++')
    print(NABELdata.head(2))
    print(NABELdata.columns)
    
    return NABELdata


def get_meteo_data(workstation_Flag: bool = False):
    # meteo data
    if workstation_Flag:
        meteopath = '../code/dataset/meteo/meteo/'
    else:
        meteopath = '../code/pncEstimator-main/data/meteo/meteo/'
    meteodata = pd.read_csv(meteopath + 'meteodata.txt', sep=';', dtype={1: str})
    meteodata.columns = ['stn', 'time', 'Radiation[W/m2] meteo', 'Temperature meteo', 'Precipitation[mm] meteo',
                        'Relative humidity[%] meteo', 'Wind speed[m/s] meteo', 'trafficVol']
    meteodata['time'] = meteodata['time'].str.slice(0, 4) + '-' + meteodata['time'].str.slice(4, 6) + \
                        '-' + meteodata['time'].str.slice(6, 8) + ' ' + meteodata['time'].str.slice(8, 10)+':00'
    meteodata['stn'] = meteodata['stn'].replace({'NABRIG': 'RIG'})
    meteodata['stn'] = meteodata['stn'].replace({'NABBER': 'BER'})
    meteodata['stn'] = meteodata['stn'].replace({'NABHAE': 'HAE'})
    meteodata['time'] = pd.to_datetime(meteodata['time'], format='%Y-%m-%d %H:%M',errors='ignore')
    print('++++++++++++++++++meteodata++++++++++++++++++')
    print(meteodata.head(2))
    return meteodata

def get_merge_data(workstation_Flag: bool = False):
    NABELdata = get_NABEL_data(workstation_Flag)
    meteodata = get_meteo_data(workstation_Flag)
    # merge data
    pmdata = pd.merge(NABELdata, meteodata, how='inner', left_on=['station', 'Date/time'],
                      right_on=['stn', 'time'])
    pmdata = pmdata[['Date/time', 'station', 'PM2.5 [ug/m3]', 'PM10 [ug/m3]',
                     'NOX [ug/m3 eq. NO2]', 'NO2 [ug/m3]',
                     'NO2/NOX ratio', 'O3 [ug/m3]',
                     'PM2.5/PM10 ratio',
                     'Radiation[W/m2] meteo', 'RAD [W/m2]', 'Temperature meteo', 'TEMP [C]',
                     'Precipitation[mm] meteo', 'PREC [mm]',
                     'Relative humidity[%] meteo', 'Wind speed[m/s] meteo', 'trafficVol']]
    pmdata['hour'] = pmdata['Date/time'].dt.hour
    pmdata['month'] = pmdata['Date/time'].dt.month
    pmdata['weekday'] = pmdata['Date/time'].dt.dayofweek
    pmdata = pmdata[(pmdata['Date/time']>='2016-01-01 01:00')&(pmdata['Date/time']<'2021-01-01 01:00')]
    pmdata = pmdata.reset_index(drop=True)
    pmdata_washed = pmdata.dropna(axis=0, how='any')
    pmdata_washed = pmdata_washed.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    return pmdata_washed

def save_feature(workstation_Flag: bool = False):
    pmdata_washed = get_merge_data(workstation_Flag)
    pmdata_washed = pmdata_washed[pmdata_washed[['PM2.5 [ug/m3]', 'PM10 [ug/m3]',
                                                 'NOX [ug/m3 eq. NO2]', 'NO2 [ug/m3]',
                                                 'O3 [ug/m3]']].applymap(lambda x: x >= 0).all(axis=1)]
    pmdata2016_2019 = pmdata_washed[(pmdata_washed['Date/time']>='2016-01-01 01:00')&(pmdata_washed['Date/time']<'2020-01-01 01:00')]
    pmdata2020 = pmdata_washed[(pmdata_washed['Date/time']>='2020-01-01 01:00')&(pmdata_washed['Date/time']<'2021-01-01 01:00')]
    pmdata2016_2019 = pmdata2016_2019.reset_index(drop=True)
    pmdata2020 = pmdata2020.reset_index(drop=True)
    if workstation_Flag:
        save_feature_file = '../code/dataset/NABEL/feature_data/'
    else:
        save_feature_file = '../code/pncEstimator-main/data/NABEL/feature_eng/'
    pmdata2016_2019.to_csv(save_feature_file + '/feature_data_2016_2019_PM.csv', index=False)
    pmdata2020.to_csv(save_feature_file + '/feature_data_2020_PM.csv', index=False)


def main():
    workstation_Flag = False
    save_feature(workstation_Flag)
    print('Feature data saved successfully!')

if __name__ == "__main__":
    main()
