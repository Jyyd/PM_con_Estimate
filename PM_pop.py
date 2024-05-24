'''
Author: jyyd23@mails.tsinghua.edu.cn
Date: 2024-05-23 17:07:59
LastEditors: jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-25 00:48:48
FilePath: PM_pop.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import numpy as np
import pandas as pd
import os, sys
os.chdir(sys.path[0])
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
from scipy.interpolate import griddata
import geopandas as gpd
import matplotlib.ticker as ticker
from scipy.io import savemat, loadmat
from scipy import interpolate
from tqdm import tqdm, trange
import rasterio
from rasterio import features
from affine import Affine
from matplotlib.path import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def plot_result(pred_filename, pred_dl_filename, results_filename, Stacking_name):
    # load pred data
    pred2020 = pd.read_csv(pred_filename)
    pred2020  = pred2020.drop(['Unnamed: 0'], axis=1)

    # load DL pred data
    pred2020_DL = pd.read_csv(pred_dl_filename)
    pred2020_DL = pred2020_DL.drop(['Unnamed: 0'], axis=1)

    # load results data
    hist_plot = pd.read_csv(results_filename)
    hist_plot['RMSE'] = np.sqrt(hist_plot['mse'])

    # calculate bias
    pred_mean = np.mean(pred2020.iloc[:, 2])
    model = []
    bias = []
    for col in pred2020.filter(like='_pred'):
        pred2020[col.split('_')[0] + '_bias'] = pred2020[col] - pred_mean
        model.append(col.split('_')[0])
        bias.append(np.sum(pred2020[col.split('_')[0] + '_bias'])/len(pred2020))
    bias_df = pd.DataFrame({'model': model, 'bias': bias})
    for i in range(len(bias_df)):
        if (bias_df['model'][i]=='ada'):
            bias_df['model'][i]='adaboost'
        elif (bias_df['model'][i]=='stack'):
            bias_df['model'][i]='stacking'

    # calculate bias DL
    cpc_mean_DL = np.mean(pred2020_DL.iloc[:, 2])
    model_DL = []
    bias_DL = []
    for col in pred2020_DL.filter(like='_pred'):
        pred2020_DL[col.split('_')[0] + '_bias'] = pred2020_DL[col] - cpc_mean_DL
        model_DL.append(col.split('_')[0])
        bias_DL.append(np.sum(pred2020_DL[col.split('_')[0] + '_bias'])/len(pred2020_DL))
    bias_df_DL = pd.DataFrame({'model': model_DL, 'bias': bias_DL})

    hist_plot_data = pd.merge(hist_plot, bias_df, on='model', how='left')
    hist_plot_data = pd.merge(hist_plot_data, bias_df_DL, on='model', how='left')
    hist_plot_data['bias'] = hist_plot_data['bias_x'].combine_first(hist_plot_data['bias_y'])
    hist_plot_data = hist_plot_data.drop(columns=['bias_x', 'bias_y'])
    hist_plot_data = hist_plot_data.iloc[:-1, :]
    hist_plot_data['model'] = ['Linear', 'Lasso', 'KNN', 'Decsion Tree', 'Random Forest',
                           'AdaBoost', 'Gradient Boosting', 'LightGBM', Stacking_name, 'RNN', 'CNN', 'LSTM']
    return hist_plot_data

def hist_plot(hist_plot_data, Stacking_name):
    Linear_Models = ['Linear', 'Lasso']
    Neighbor_based_Models = ['KNN']
    Tree_based_Models = ['Decsion Tree', 'Random Forest','AdaBoost', 'Gradient Boosting', 'LightGBM']
    Stem_PNC = [Stacking_name]
    Deep_Learning_Models = ['RNN', 'CNN', 'LSTM']
    
    hist_plot_data['model_type'] = np.where(hist_plot_data['model'].isin(Linear_Models), 'Linear Models',
                                                    np.where(hist_plot_data['model'].isin(Neighbor_based_Models), 'Neighbor-based Models',
                                                                np.where(hist_plot_data['model'].isin(Tree_based_Models), 'Tree-based Models',
                                                                            np.where(hist_plot_data['model'].isin(Stem_PNC), Stacking_name,
                                                                                        np.where(hist_plot_data['model'].isin(Deep_Learning_Models),
                                                                                                 'Deep Learning Models', 'Other')))))
    model_type_means = hist_plot_data.groupby('model_type')[['evs', 'mae', 'mse', 'r2', 'RMSE', 'bias']].mean().reset_index()
    model_type_means['model_type'] = pd.Categorical(model_type_means['model_type'], ['Linear Models', 
                                                                                    'Neighbor-based Models', 'Tree-based Models',
                                                                                    'Deep Learning Models', Stacking_name])
    model_type_means = model_type_means.sort_values('model_type')
    model_type_var = hist_plot_data.groupby('model_type')[['evs', 'mae', 'mse', 'r2', 'RMSE', 'bias']].var().reset_index()
    model_type_var['model_type'] = pd.Categorical(model_type_var['model_type'], ['Linear Models', 
                                                                                    'Neighbor-based Models', 'Tree-based Models',
                                                                                    'Deep Learning Models', Stacking_name])
    model_type_var = model_type_var.sort_values('model_type')
    model_type_std = hist_plot_data.groupby('model_type')[['evs', 'mae', 'mse', 'r2', 'RMSE', 'bias']].std().reset_index()
    model_type_std['model_type'] = pd.Categorical(model_type_std['model_type'], ['Linear Models',
                                                                                    'Neighbor-based Models', 'Tree-based Models',
                                                                                    'Deep Learning Models', Stacking_name])
    model_type_std = model_type_std.sort_values('model_type')
    return model_type_means, model_type_var, model_type_std



def get_tif_data(tif_file):
    ds = xr.open_rasterio(tif_file)
    tiflon = np.array(ds['x'])
    tiflat = np.array(ds['y'])
    tifdata = np.array(ds.values[0])/255
    tifdata = np.ma.masked_where(tifdata == 1, tifdata)
    return tiflon, tiflat, tifdata

def get_mat_data(mat_file):
    pmmat_data = loadmat(mat_file)
    matlon = pmmat_data['lonNew'][0][:]
    matlat = pmmat_data['latNew'][0][:]
    avgPM = pmmat_data['avgConc']
    pop_data = pmmat_data['interpolated_pop']
    new_matlon = np.linspace(matlon.min(), matlon.max(), 531)
    new_matlat = np.linspace(matlat.min(), matlat.max(), 211)
    new_matlat = new_matlat[::-1]
    return new_matlon, new_matlat, avgPM, pop_data

def resample_to_mat_grid(weighted_avg, mat_lon, mat_lat, tif_lon, tif_lat):
    resampled_grid = np.zeros((len(mat_lat), len(mat_lon)))
    
    tif_lon_res = np.abs(tif_lon[1] - tif_lon[0])
    tif_lat_res = np.abs(tif_lat[1] - tif_lat[0])
    mat_lon_res = np.abs(mat_lon[1] - mat_lon[0])
    mat_lat_res = np.abs(mat_lat[1] - mat_lat[0])
    
    for i in trange(len(mat_lat)):
        for j in range(len(mat_lon)):
            lon_mask = (tif_lon >= mat_lon[j] - mat_lon_res/2) & (tif_lon < mat_lon[j] + mat_lon_res/2)
            lat_mask = (tif_lat >= mat_lat[i] - mat_lat_res/2) & (tif_lat < mat_lat[i] + mat_lat_res/2)
            weighted_avg_sub = weighted_avg[np.ix_(lat_mask, lon_mask)]
            
            if weighted_avg_sub.size > 0:
                resampled_grid[i, j] = np.mean(weighted_avg_sub) * weighted_avg_sub.size

    resample_land = np.nan_to_num(resampled_grid)
    resample_land_normalized = (resample_land - np.amin(resample_land)) / (np.amax(resample_land) - np.amin(resample_land))

    return resample_land_normalized

def get_land_mat():
    tiflon, tiflat, BuiltUp_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_BuiltUp_CoverFraction.tif')
    _, _, Bare_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_Bare_CoverFraction.tif')
    _, _, Crops_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_Crops_CoverFraction.tif')
    _, _, Grass_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_Grass_CoverFraction.tif')
    _, _, Shrub_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_Shrub_CoverFraction.tif')
    _, _, MossLichen_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_MossLichen_CoverFraction.tif')
    _, _, Snow_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_Snow_CoverFraction.tif')
    _, _, PermanentWater_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_PermanentWater_CoverFraction.tif')
    _, _, SeasonalWater_tifdata = get_tif_data('./tif_file/wash_tiff/switzerland_SeasonalWater_CoverFraction.tif')
    print(Bare_tifdata.shape, tiflat.shape, tiflon.shape)
    matlon, matlat, avgPM25, pop_all = get_mat_data('./out/mat_file/PMpopmat/PM25_all_data.mat')
    _, _, avgPM10, _ = get_mat_data('./out/mat_file/PMpopmat/PM10_all_data.mat')

    resample_land_BuiltUp = resample_to_mat_grid(BuiltUp_tifdata, matlon, matlat, tiflon, tiflat)
    resample_land_Bare = resample_to_mat_grid(Bare_tifdata, matlon, matlat, tiflon, tiflat)
    resample_land_Crops = resample_to_mat_grid(Crops_tifdata, matlon, matlat, tiflon, tiflat)
    resample_land_Grass = resample_to_mat_grid(Grass_tifdata, matlon, matlat, tiflon, tiflat)
    resample_land_Shrub = resample_to_mat_grid(Shrub_tifdata, matlon, matlat, tiflon, tiflat)
    resample_land_MossLichen = resample_to_mat_grid(MossLichen_tifdata, matlon, matlat, tiflon, tiflat)
    resample_land_Snow = resample_to_mat_grid(Snow_tifdata, matlon, matlat, tiflon, tiflat)
    resample_land_PermanentWater = resample_to_mat_grid(PermanentWater_tifdata, matlon, matlat, tiflon, tiflat)
    resample_land_SeasonalWater = resample_to_mat_grid(SeasonalWater_tifdata, matlon, matlat, tiflon, tiflat)
    mat_data = {'avgPM25': avgPM25, 'avgPM10': avgPM10,
                'matlon': matlon, 'matlat': matlat,
                'BuiltUp': resample_land_BuiltUp, 'Bare': resample_land_Bare,
                'Crops': resample_land_Crops, 'Grass': resample_land_Grass,
                'Shrub': resample_land_Shrub, 'MossLichen': resample_land_MossLichen,
                'Snow': resample_land_Snow, 'PermanentWater': resample_land_PermanentWater,
                'SeasonalWater': resample_land_SeasonalWater}
    # savemat('./out/mat_file/landmat/land_pm.mat', mat_data)


def load_pm_pop_data(pm_type:str='PM10'):
    if pm_type == 'PM10':
        file_path = './out/mat_file/PMpopmat/PM10_pop_all.mat'
        avgConc_name = 'avgConc_pm10'
    elif pm_type == 'PM2.5':
        file_path = './out/mat_file/PMpopmat/PM25_pop_all.mat'
        avgConc_name = 'avgConc_pm25'
    else:
        raise ValueError('Invalid PM type. Please choose between PM10 and PM2.5')

    pmmat_data = loadmat(file_path)
    matlon = pmmat_data['matlon'][0][:]
    matlat = pmmat_data['matlat'][0][:]
    avgPM = pmmat_data[avgConc_name]
    pop_bt = pmmat_data['pop_bt']
    pop_mt = pmmat_data['pop_mt']
    pop_ft = pmmat_data['pop_ft']
    pop_bt_0_14 = pmmat_data['pop_0_14']
    pop_bt_15_64 = pmmat_data['pop_15_64']
    pop_bt_65 = pmmat_data['pop_65']

    return (matlon, matlat, avgPM,
            pop_bt, pop_mt, pop_ft,
            pop_bt_0_14, pop_bt_15_64, pop_bt_65)

def save_tiff(save_tiff_name:str, save_tiff_data, lon_grid, lat_grid):
    transform = Affine.translation(lon_grid[0][0], lat_grid[0][0]) * Affine.scale(lon_grid[0,1]-lon_grid[0,0],
                                                                              lat_grid[1,0]-lat_grid[0,0])
    dataset = rasterio.open(
        save_tiff_name, 'w',
        driver='GTiff',
        height=lon_grid.shape[0],
        width=lat_grid.shape[1],
        count=1,
        dtype=save_tiff_data.dtype,
        crs='+proj=latlong',
        transform=transform,
    )
    dataset.write(save_tiff_data, 1)
    dataset.close()

def get_district_data(tiff_filename:str, value_name:str, data):
    tiff_dataset = rasterio.open(tiff_filename)
    gdf = gpd.read_file('../code/pncEstimator-main/data/geoshp/gadm36_CHE_3.shp')
    
    weighted_data_by_district_mean = {}
    weighted_data_by_district_sum = {}
    weighted_data_by_district = {}
    for idx, row in gdf.iterrows():
        district_name = row['NAME_3']
        district_shape = row['geometry']

        mask = features.geometry_mask([district_shape], transform=tiff_dataset.transform, invert=True,
                                    out_shape=(tiff_dataset.height, tiff_dataset.width), all_touched=True)

        district_value = data[mask]
        weighted_data_by_district_sum[district_name] = np.sum(district_value)
        weighted_data_by_district_mean[district_name] = np.mean(district_value)
        weighted_data_by_district[district_name] = district_value
    
    weighted_district_mean = []
    weight_value_mean = []
    weight_value_sum = []
    for district, value in weighted_data_by_district_mean.items():
        weighted_district_mean.append(district)
        weight_value_mean.append(value)
    for district, value in weighted_data_by_district_sum.items():
        weight_value_sum.append(value)
    for district, value in weighted_data_by_district.items():
        weighted_data_by_district[district] = value
    
    tiff_dataset.close()

    weighted_data = pd.DataFrame(weighted_district_mean, columns=['District'])
    weighted_data[value_name + '_mean'] = weight_value_mean
    weighted_data[value_name + '_sum'] = weight_value_sum
    
    return weighted_data, weighted_data_by_district

def get_district_data1(tiff_filename:str, value_name:str, data):
    tiff_dataset = rasterio.open(tiff_filename)
    gdf = gpd.read_file('../code/pncEstimator-main/data/geoshp/gadm36_CHE_1.shp')
    
    weighted_data_by_district_mean = {}
    weighted_data_by_district_sum = {}
    weighted_data_by_district = {}
    for idx, row in gdf.iterrows():
        district_name = row['NAME_1']
        district_shape = row['geometry']

        mask = features.geometry_mask([district_shape], transform=tiff_dataset.transform, invert=True,
                                    out_shape=(tiff_dataset.height, tiff_dataset.width), all_touched=True)

        district_value = data[mask]
        weighted_data_by_district_sum[district_name] = np.sum(district_value)
        weighted_data_by_district_mean[district_name] = np.mean(district_value)
        weighted_data_by_district[district_name] = district_value
    
    weighted_district_mean = []
    weight_value_mean = []
    weight_value_sum = []
    for district, value in weighted_data_by_district_mean.items():
        weighted_district_mean.append(district)
        weight_value_mean.append(value)
    for district, value in weighted_data_by_district_sum.items():
        weight_value_sum.append(value)
    for district, value in weighted_data_by_district.items():
        weighted_data_by_district[district] = value
    
    tiff_dataset.close()

    weighted_data = pd.DataFrame(weighted_district_mean, columns=['District'])
    weighted_data[value_name + '_mean'] = weight_value_mean
    weighted_data[value_name + '_sum'] = weight_value_sum
    
    return weighted_data, weighted_data_by_district

def save_annual_tiff():
    (matlon, matlat, avgPM10, 
    pop_bt, pop_mt, pop_ft, pop_bt_0_14, pop_bt_15_64, pop_bt_65) = load_pm_pop_data('PM10')
    (matlon, matlat, avgPM25, 
    pop_bt, pop_mt, pop_ft, pop_bt_0_14, pop_bt_15_64, pop_bt_65) = load_pm_pop_data('PM2.5')

    daily_pm10 = np.load('./out/npy_file/daily_pm10.npy')
    daily_pm25 = np.load('./out/npy_file/daily_pm25.npy')
    aqg_pm10 = np.zeros_like(daily_pm10)
    aqg_pm25 = np.zeros_like(daily_pm25)
    aqg_pm10[daily_pm10 >= 45] = 1
    aqg_pm25[daily_pm25 >= 15] = 1
    aqg_pm10 = np.sum(aqg_pm10, axis=0)
    aqg_pm25 = np.sum(aqg_pm25, axis=0)

    pm252020_data  = np.nan_to_num(avgPM25, nan=0.0)
    pm102020_data  = np.nan_to_num(avgPM10, nan=0.0)
    lon_grid, lat_grid = np.meshgrid(matlon, matlat)
    save_tiff('./tif_file/annual/pm25_2020.tif', pm252020_data, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pm10_2020.tif', pm102020_data, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_bt.tif', pop_bt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_mt.tif', pop_mt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_ft.tif', pop_ft, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_bt_0_14.tif', pop_bt_0_14, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_bt_15_64.tif', pop_bt_15_64, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_bt_65.tif', pop_bt_65, lon_grid, lat_grid)

    save_tiff('./tif_file/annual/aqg_pm10.tif', aqg_pm10, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/aqg_pm25.tif', aqg_pm25, lon_grid, lat_grid)

    save_tiff('./tif_file/annual/pm25_pop.tif', pm252020_data*pop_bt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pm10_pop.tif', pm102020_data*pop_bt, lon_grid, lat_grid)

    weighted_pm25, weighted_pm25_district = get_district_data('./tif_file/annual/pm25_2020.tif', 'pm25_2020', pm252020_data)
    weighted_pm10, weighted_pm10_district = get_district_data('./tif_file/annual/pm10_2020.tif', 'pm10_2020', pm102020_data)

    weighted_daily_pm25, weighted_daily_pm25_district = get_district_data('./tif_file/annual/aqg_pm25.tif', 'day_pm25', aqg_pm25)
    weighted_daily_pm10, weighted_daily_pm10_district = get_district_data('./tif_file/annual/aqg_pm10.tif', 'day_pm10', aqg_pm10)

    weighted_pop_bt, weighted_pop_bt_district = get_district_data('./tif_file/annual/pop_bt.tif', 'pop_bt', pop_bt)
    weighted_pop_mt, weighted_pop_mt_district = get_district_data('./tif_file/annual/pop_mt.tif', 'pop_mt', pop_mt)
    weighted_pop_ft, weighted_pop_ft_district = get_district_data('./tif_file/annual/pop_ft.tif', 'pop_ft', pop_ft)
    weighted_pop_bt_0_14, weighted_pop_bt_0_14_district = get_district_data('./tif_file/annual/pop_bt_0_14.tif', 'pop_bt_0_14', pop_bt_0_14)
    weighted_pop_bt_15_64, weighted_pop_bt_15_64_district = get_district_data('./tif_file/annual/pop_bt_15_64.tif', 'pop_bt_15_64', pop_bt_15_64)
    weighted_pop_bt_65, weighted_pop_bt_65_district = get_district_data('./tif_file/annual/pop_bt_65.tif', 'pop_bt_65', pop_bt_65)

    weighted_pop_bt_pm25, _ = get_district_data('./tif_file/annual/pm25_pop.tif', 'pm25_pop', pm252020_data*pop_bt)
    weighted_pop_bt_pm10, _ = get_district_data('./tif_file/annual/pm10_pop.tif', 'pm10_pop', pm102020_data*pop_bt)

    dataframes = [weighted_pm25, weighted_pm10,
                  weighted_pop_bt_pm25, weighted_pop_bt_pm10,
                  weighted_daily_pm25, weighted_daily_pm10,
                  weighted_pop_bt, weighted_pop_mt, weighted_pop_ft,
                  weighted_pop_bt_0_14, weighted_pop_bt_15_64, weighted_pop_bt_65]
    weighted_data_all = dataframes[0]

    for df in dataframes[1:]:
        weighted_data_all = pd.merge(weighted_data_all, df, on='District')

    ajdusted_pop = np.sum(pop_bt)/np.sum(weighted_data_all['pop_bt_mean'])
    ajdusted_pop_mt = np.sum(pop_mt)/np.sum(weighted_data_all['pop_mt_mean'])
    ajdusted_pop_ft = np.sum(pop_ft)/np.sum(weighted_data_all['pop_ft_mean'])
    ajdusted_pop_bt_0_14 = np.sum(pop_bt_0_14)/np.sum(weighted_data_all['pop_bt_0_14_mean'])
    ajdusted_pop_bt_15_64 = np.sum(pop_bt_15_64)/np.sum(weighted_data_all['pop_bt_15_64_mean'])
    ajdusted_pop_bt_65 = np.sum(pop_bt_65)/np.sum(weighted_data_all['pop_bt_65_mean'])
    weighted_data_all['pop_bt_mean'] = weighted_data_all['pop_bt_mean']*ajdusted_pop
    weighted_data_all['pop_mt_mean'] = weighted_data_all['pop_mt_mean']*ajdusted_pop_mt
    weighted_data_all['pop_ft_mean'] = weighted_data_all['pop_ft_mean']*ajdusted_pop_ft
    weighted_data_all['pop_bt_0_14_mean'] = weighted_data_all['pop_bt_0_14_mean']*ajdusted_pop_bt_0_14
    weighted_data_all['pop_bt_15_64_mean'] = weighted_data_all['pop_bt_15_64_mean']*ajdusted_pop_bt_15_64
    weighted_data_all['pop_bt_65_mean'] = weighted_data_all['pop_bt_65_mean']*ajdusted_pop_bt_65

    return weighted_data_all

def save_annual_tiff1():
    (matlon, matlat, avgPM10, 
    pop_bt, pop_mt, pop_ft, pop_bt_0_14, pop_bt_15_64, pop_bt_65) = load_pm_pop_data('PM10')
    (matlon, matlat, avgPM25, 
    pop_bt, pop_mt, pop_ft, pop_bt_0_14, pop_bt_15_64, pop_bt_65) = load_pm_pop_data('PM2.5')

    daily_pm10 = np.load('./out/npy_file/daily_pm10.npy')
    daily_pm25 = np.load('./out/npy_file/daily_pm25.npy')
    aqg_pm10 = np.zeros_like(daily_pm10)
    aqg_pm25 = np.zeros_like(daily_pm25)
    aqg_pm10[daily_pm10 >= 45] = 1
    aqg_pm25[daily_pm25 >= 15] = 1
    aqg_pm10 = np.sum(aqg_pm10, axis=0)
    aqg_pm25 = np.sum(aqg_pm25, axis=0)

    pm252020_data  = np.nan_to_num(avgPM25, nan=0.0)
    pm102020_data  = np.nan_to_num(avgPM10, nan=0.0)
    lon_grid, lat_grid = np.meshgrid(matlon, matlat)
    save_tiff('./tif_file/annual1/pm25_2020.tif', pm252020_data, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/pm10_2020.tif', pm102020_data, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/pop_bt.tif', pop_bt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/pop_mt.tif', pop_mt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/pop_ft.tif', pop_ft, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/pop_bt_0_14.tif', pop_bt_0_14, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/pop_bt_15_64.tif', pop_bt_15_64, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/pop_bt_65.tif', pop_bt_65, lon_grid, lat_grid)

    save_tiff('./tif_file/annual1/aqg_pm10.tif', aqg_pm10, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/aqg_pm25.tif', aqg_pm25, lon_grid, lat_grid)

    save_tiff('./tif_file/annual1/pm25_pop.tif', pm252020_data*pop_bt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual1/pm10_pop.tif', pm102020_data*pop_bt, lon_grid, lat_grid)

    weighted_pm25, weighted_pm25_district = get_district_data1('./tif_file/annual1/pm25_2020.tif', 'pm25_2020', pm252020_data)
    weighted_pm10, weighted_pm10_district = get_district_data1('./tif_file/annual1/pm10_2020.tif', 'pm10_2020', pm102020_data)

    weighted_daily_pm25, weighted_daily_pm25_district = get_district_data1('./tif_file/annual1/aqg_pm25.tif', 'day_pm25', aqg_pm25)
    weighted_daily_pm10, weighted_daily_pm10_district = get_district_data1('./tif_file/annual1/aqg_pm10.tif', 'day_pm10', aqg_pm10)

    weighted_pop_bt, weighted_pop_bt_district = get_district_data1('./tif_file/annual1/pop_bt.tif', 'pop_bt', pop_bt)
    weighted_pop_mt, weighted_pop_mt_district = get_district_data1('./tif_file/annual1/pop_mt.tif', 'pop_mt', pop_mt)
    weighted_pop_ft, weighted_pop_ft_district = get_district_data1('./tif_file/annual1/pop_ft.tif', 'pop_ft', pop_ft)
    weighted_pop_bt_0_14, weighted_pop_bt_0_14_district = get_district_data1('./tif_file/annual1/pop_bt_0_14.tif', 'pop_bt_0_14', pop_bt_0_14)
    weighted_pop_bt_15_64, weighted_pop_bt_15_64_district = get_district_data1('./tif_file/annual1/pop_bt_15_64.tif', 'pop_bt_15_64', pop_bt_15_64)
    weighted_pop_bt_65, weighted_pop_bt_65_district = get_district_data1('./tif_file/annual1/pop_bt_65.tif', 'pop_bt_65', pop_bt_65)

    weighted_pop_bt_pm25, _ = get_district_data1('./tif_file/annual1/pm25_pop.tif', 'pm25_pop', pm252020_data*pop_bt)
    weighted_pop_bt_pm10, _ = get_district_data1('./tif_file/annual1/pm10_pop.tif', 'pm10_pop', pm102020_data*pop_bt)

    dataframes = [weighted_pm25, weighted_pm10,
                  weighted_pop_bt_pm25, weighted_pop_bt_pm10,
                  weighted_daily_pm25, weighted_daily_pm10,
                  weighted_pop_bt, weighted_pop_mt, weighted_pop_ft,
                  weighted_pop_bt_0_14, weighted_pop_bt_15_64, weighted_pop_bt_65]
    weighted_data_all = dataframes[0]

    for df in dataframes[1:]:
        weighted_data_all = pd.merge(weighted_data_all, df, on='District')

    ajdusted_pop = np.sum(pop_bt)/np.sum(weighted_data_all['pop_bt_mean'])
    ajdusted_pop_mt = np.sum(pop_mt)/np.sum(weighted_data_all['pop_mt_mean'])
    ajdusted_pop_ft = np.sum(pop_ft)/np.sum(weighted_data_all['pop_ft_mean'])
    ajdusted_pop_bt_0_14 = np.sum(pop_bt_0_14)/np.sum(weighted_data_all['pop_bt_0_14_mean'])
    ajdusted_pop_bt_15_64 = np.sum(pop_bt_15_64)/np.sum(weighted_data_all['pop_bt_15_64_mean'])
    ajdusted_pop_bt_65 = np.sum(pop_bt_65)/np.sum(weighted_data_all['pop_bt_65_mean'])
    weighted_data_all['pop_bt_mean'] = weighted_data_all['pop_bt_mean']*ajdusted_pop
    weighted_data_all['pop_mt_mean'] = weighted_data_all['pop_mt_mean']*ajdusted_pop_mt
    weighted_data_all['pop_ft_mean'] = weighted_data_all['pop_ft_mean']*ajdusted_pop_ft
    weighted_data_all['pop_bt_0_14_mean'] = weighted_data_all['pop_bt_0_14_mean']*ajdusted_pop_bt_0_14
    weighted_data_all['pop_bt_15_64_mean'] = weighted_data_all['pop_bt_15_64_mean']*ajdusted_pop_bt_15_64
    weighted_data_all['pop_bt_65_mean'] = weighted_data_all['pop_bt_65_mean']*ajdusted_pop_bt_65

    return weighted_data_all

def save_annual_tiff_land():
    (matlon, matlat, avgPM10, 
    pop_bt, pop_mt, pop_ft, pop_bt_0_14, pop_bt_15_64, pop_bt_65) = load_pm_pop_data('PM10')
    (matlon, matlat, avgPM25, 
    pop_bt, pop_mt, pop_ft, pop_bt_0_14, pop_bt_15_64, pop_bt_65) = load_pm_pop_data('PM2.5')

    daily_pm10 = np.load('./out/npy_file/daily_pm10.npy')
    daily_pm25 = np.load('./out/npy_file/daily_pm25.npy')
    aqg_pm10 = np.zeros_like(daily_pm10)
    aqg_pm25 = np.zeros_like(daily_pm25)
    aqg_pm10[daily_pm10 >= 45] = 1
    aqg_pm25[daily_pm25 >= 15] = 1
    aqg_pm10 = np.sum(aqg_pm10, axis=0)
    aqg_pm25 = np.sum(aqg_pm25, axis=0)

    pm252020_data  = np.nan_to_num(avgPM25, nan=0.0)
    pm102020_data  = np.nan_to_num(avgPM10, nan=0.0)
    lon_grid, lat_grid = np.meshgrid(matlon, matlat)

    land_pm = loadmat('./out/mat_file/landmat/land_pm.mat')
    BuiltUp = land_pm['BuiltUp']
    Bare = land_pm['Bare']
    Tree = land_pm['Tree']
    Crops = land_pm['Crops']
    Grass = land_pm['Grass']
    Shrub = land_pm['Shrub']
    MossLichen = land_pm['MossLichen']
    Snow = land_pm['Snow']
    PermanentWater = land_pm['PermanentWater']
    SeasonalWater = land_pm['SeasonalWater']

    save_tiff('./tif_file/annual/pm25_2020.tif', pm252020_data, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pm10_2020.tif', pm102020_data, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_bt.tif', pop_bt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_mt.tif', pop_mt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_ft.tif', pop_ft, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_bt_0_14.tif', pop_bt_0_14, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_bt_15_64.tif', pop_bt_15_64, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pop_bt_65.tif', pop_bt_65, lon_grid, lat_grid)

    save_tiff('./tif_file/annual/aqg_pm10.tif', aqg_pm10, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/aqg_pm25.tif', aqg_pm25, lon_grid, lat_grid)

    save_tiff('./tif_file/annual/pm25_pop.tif', pm252020_data*pop_bt, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/pm10_pop.tif', pm102020_data*pop_bt, lon_grid, lat_grid)

    save_tiff('./tif_file/annual/BuiltUp.tif', BuiltUp, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/Bare.tif', Bare, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/Tree.tif', Tree, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/Crops.tif', Crops, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/Grass.tif', Grass, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/Shrub.tif', Shrub, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/MossLichen.tif', MossLichen, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/Snow.tif', Snow, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/PermanentWater.tif', PermanentWater, lon_grid, lat_grid)
    save_tiff('./tif_file/annual/SeasonalWater.tif', SeasonalWater, lon_grid, lat_grid)

    weighted_pm25, weighted_pm25_district = get_district_data('./tif_file/annual/pm25_2020.tif', 'pm25_2020', pm252020_data)
    weighted_pm10, weighted_pm10_district = get_district_data('./tif_file/annual/pm10_2020.tif', 'pm10_2020', pm102020_data)

    weighted_daily_pm25, weighted_daily_pm25_district = get_district_data('./tif_file/annual/aqg_pm25.tif', 'day_pm25', aqg_pm25)
    weighted_daily_pm10, weighted_daily_pm10_district = get_district_data('./tif_file/annual/aqg_pm10.tif', 'day_pm10', aqg_pm10)

    weighted_pop_bt, weighted_pop_bt_district = get_district_data('./tif_file/annual/pop_bt.tif', 'pop_bt', pop_bt)
    weighted_pop_mt, weighted_pop_mt_district = get_district_data('./tif_file/annual/pop_mt.tif', 'pop_mt', pop_mt)
    weighted_pop_ft, weighted_pop_ft_district = get_district_data('./tif_file/annual/pop_ft.tif', 'pop_ft', pop_ft)
    weighted_pop_bt_0_14, weighted_pop_bt_0_14_district = get_district_data('./tif_file/annual/pop_bt_0_14.tif', 'pop_bt_0_14', pop_bt_0_14)
    weighted_pop_bt_15_64, weighted_pop_bt_15_64_district = get_district_data('./tif_file/annual/pop_bt_15_64.tif', 'pop_bt_15_64', pop_bt_15_64)
    weighted_pop_bt_65, weighted_pop_bt_65_district = get_district_data('./tif_file/annual/pop_bt_65.tif', 'pop_bt_65', pop_bt_65)

    weighted_pop_bt_pm25, _ = get_district_data('./tif_file/annual/pm25_pop.tif', 'pm25_pop', pm252020_data*pop_bt)
    weighted_pop_bt_pm10, _ = get_district_data('./tif_file/annual/pm10_pop.tif', 'pm10_pop', pm102020_data*pop_bt)

    weighted_BuiltUp, _ = get_district_data('./tif_file/annual/BuiltUp.tif', 'BuiltUp', BuiltUp)
    weighted_Bare, _ = get_district_data('./tif_file/annual/Bare.tif', 'Bare', Bare)
    weighted_Tree, _ = get_district_data('./tif_file/annual/Tree.tif', 'Tree', Tree)
    weighted_Crops, _ = get_district_data('./tif_file/annual/Crops.tif', 'Crops', Crops)
    weighted_Grass, _ = get_district_data('./tif_file/annual/Grass.tif', 'Grass', Grass)
    weighted_Shrub, _ = get_district_data('./tif_file/annual/Shrub.tif', 'Shrub', Shrub)
    weighted_MossLichen, _ = get_district_data('./tif_file/annual/MossLichen.tif', 'MossLichen', MossLichen)
    weighted_Snow, _ = get_district_data('./tif_file/annual/Snow.tif', 'Snow', Snow)
    weighted_PermanentWater, _ = get_district_data('./tif_file/annual/PermanentWater.tif', 'PermanentWater', PermanentWater)
    weighted_SeasonalWater, _ = get_district_data('./tif_file/annual/SeasonalWater.tif', 'SeasonalWater', SeasonalWater)


    dataframes = [weighted_pm25, weighted_pm10,
                  weighted_pop_bt_pm25, weighted_pop_bt_pm10,
                  weighted_daily_pm25, weighted_daily_pm10,
                  weighted_pop_bt, weighted_pop_mt, weighted_pop_ft,
                  weighted_pop_bt_0_14, weighted_pop_bt_15_64, weighted_pop_bt_65, 
                  weighted_BuiltUp, weighted_Bare, weighted_Tree, weighted_Crops,
                  weighted_Grass, weighted_Shrub, weighted_MossLichen, weighted_Snow,
                  weighted_PermanentWater, weighted_SeasonalWater]
    weighted_data_all = dataframes[0]

    for df in dataframes[1:]:
        weighted_data_all = pd.merge(weighted_data_all, df, on='District', how='outer')

    ajdusted_pop = np.sum(pop_bt)/np.sum(weighted_data_all['pop_bt_mean'])
    ajdusted_pop_mt = np.sum(pop_mt)/np.sum(weighted_data_all['pop_mt_mean'])
    ajdusted_pop_ft = np.sum(pop_ft)/np.sum(weighted_data_all['pop_ft_mean'])
    ajdusted_pop_bt_0_14 = np.sum(pop_bt_0_14)/np.sum(weighted_data_all['pop_bt_0_14_mean'])
    ajdusted_pop_bt_15_64 = np.sum(pop_bt_15_64)/np.sum(weighted_data_all['pop_bt_15_64_mean'])
    ajdusted_pop_bt_65 = np.sum(pop_bt_65)/np.sum(weighted_data_all['pop_bt_65_mean'])
    weighted_data_all['pop_bt_mean'] = weighted_data_all['pop_bt_mean']*ajdusted_pop
    weighted_data_all['pop_mt_mean'] = weighted_data_all['pop_mt_mean']*ajdusted_pop_mt
    weighted_data_all['pop_ft_mean'] = weighted_data_all['pop_ft_mean']*ajdusted_pop_ft
    weighted_data_all['pop_bt_0_14_mean'] = weighted_data_all['pop_bt_0_14_mean']*ajdusted_pop_bt_0_14
    weighted_data_all['pop_bt_15_64_mean'] = weighted_data_all['pop_bt_15_64_mean']*ajdusted_pop_bt_15_64
    weighted_data_all['pop_bt_65_mean'] = weighted_data_all['pop_bt_65_mean']*ajdusted_pop_bt_65

    return weighted_data_all

def main():
    # get the land matrix data
    get_land_mat()
    

if __name__ == '__main__':
    main()