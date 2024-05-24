'''
Author: JYYD jyyd23@mails.tsinghua.edu.cn
Date: 2024-05-21 21:16:34
LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-05-21 22:46:47
FilePath: PM_train.py
Description: 

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import time
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score,cross_val_predict, cross_validate, RandomizedSearchCV
import pylab
import warnings
warnings.filterwarnings("ignore")
from joblib import dump, load

from sklearn.linear_model import LinearRegression, Lasso
from sklearn import neighbors, svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor


    
def pred_metrics(model_fit, x_test, y_test, data_concat):

    y_pred = model_fit.predict(x_test)
    resid = y_pred - y_test

    print('----------------------------------')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print('MSE: ', mse)
    print('MAE: ', mae)
    print('r2 score: ', r2)
    print('Explained_variance: ', evs)

    pred = pd.DataFrame(pred)
    resid = pd.DataFrame(resid)
    data_pred = pd.concat([data_concat, pred, resid], axis=1)

    return data_pred

def load_pm_data(workstation_Flag: bool = False, PM_type: str = 'PM2.5'):
    if workstation_Flag:
        pm_feature_path = '../code/dataset/NABEL/feature_data/'
    else:
        pm_feature_path = '../code/pncEstimator-main/data/NABEL/feature_eng/'
    pmdata2016_2019 = pd.read_csv(pm_feature_path + 'feature_data_2016_2019_PM.csv')
    pmdata2020 = pd.read_csv(pm_feature_path + 'feature_data_2020_PM.csv')

    if PM_type == 'PM2.5':
        the_y_col_name = 'PM2.5 [ug/m3]'
        the_first_feature_col_name = 'PM10 [ug/m3]'
    elif PM_type == 'PM10':
        the_y_col_name = 'PM10 [ug/m3]'
        the_first_feature_col_name = 'PM2.5 [ug/m3]'
    else:
        raise ValueError('PM_type should be either PM2.5 or PM10')


    pmdata2016_2019_PM = pmdata2016_2019[['Date/time', 'station',
                                          the_y_col_name, the_first_feature_col_name,
                                          'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                                          'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                                          'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                                          'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday']]
    pmdata2020_PM = pmdata2020[['Date/time', 'station',
                                 the_y_col_name, the_first_feature_col_name,
                                'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                                'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                                'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                                'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday']]
    # split data and the ratio
    pm_x_train = pmdata2016_2019_PM.iloc[:,3:].values
    pm_y_train = pmdata2016_2019_PM.iloc[:,2].values
    pm_x_test = pmdata2020_PM.iloc[:,3:].values
    pm_y_test = pmdata2020_PM.iloc[:,2].values
    print('-----------------------PM2.5---------------------------')
    print(pm_y_train.shape, pm_y_test.shape)
    print(f'The train dataset ratio of {PM_type} = ', pm_y_train.shape[0] / (pm_y_test.shape[0] + pm_y_train.shape[0]))
    print(f'The test dataset ratio of {PM_type} = ', pm_y_test.shape[0] / (pm_y_test.shape[0] + pm_y_train.shape[0]))
    return pm_x_train, pm_y_train, pm_x_test, pm_y_test, pmdata2016_2019_PM, pmdata2020_PM

def stander_data(workstation_Flag: bool = False):
    scaler_std = StandardScaler()
    pm25_x_train, pm25_y_train, pm25_x_test, pm25_y_test, pmdata2016_2019_25, pmdata2020_25 = load_pm_data(workstation_Flag, 'PM2.5')
    pm10_x_train, pm10_y_train, pm10_x_test, pm10_y_test, pmdata2016_2019_10, pmdata2020_10 = load_pm_data(workstation_Flag, 'PM10')
    pm_x_all_25 = np.vstack((pm25_x_train, pm25_x_test))
    pm_x_all_scaler_25 = scaler_std.fit(pm_x_all_25)
    pm_x_train_25 = pm_x_all_scaler_25.transform(pm25_x_train)
    pm_x_test_25 = pm_x_all_scaler_25.transform(pm25_x_test)
    pm_x_all_10 = np.vstack((pm10_x_train, pm10_x_test))
    pm_x_all_scaler_10 = scaler_std.fit(pm_x_all_10)
    pm_x_train_10 = pm_x_all_scaler_10.transform(pm10_x_train)
    pm_x_test_10 = pm_x_all_scaler_10.transform(pm10_x_test)

    return (pm_x_all_scaler_25, pm_x_all_scaler_10,
            pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
            pmdata2016_2019_25, pmdata2020_25,
            pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
            pmdata2016_2019_10, pmdata2020_10)

def train_model(workstation_Flag):
    (_, _, pm_x_train_25, pm25_y_train, pm_x_test_25, pm25_y_test,
     pmdata2016_2019_25, pmdata2020_25,
     pm_x_train_10, pm10_y_train, pm_x_test_10, pm10_y_test,
     pmdata2016_2019_10, pmdata2020_10) = stander_data(workstation_Flag)
    x_train_25 = pm_x_train_25
    y_train_25 = pm25_y_train
    x_test_25 = pm_x_test_25
    y_test_25 = pm25_y_test
    linear_reg_pm25 = LinearRegression().fit(x_train_25, y_train_25)
    lasso_reg_pm25 = Lasso(alpha=0.1).fit(x_train_25, y_train_25)
    knn_reg_pm25 = neighbors.KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=20).fit(x_train_25, y_train_25)
    tree_reg_pm25 = DecisionTreeRegressor(random_state=42, max_depth=30, criterion='squared_error').fit(x_train_25, y_train_25)
    rf_reg_pm25 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=20).fit(x_train_25, y_train_25)
    ada_reg_pm25 = AdaBoostRegressor(random_state=42, n_estimators=50).fit(x_train_25, y_train_25)
    gbr_reg_pm25 = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.05).fit(x_train_25, y_train_25)
    lgb_reg_pm25 = lgb.LGBMRegressor(random_state=42, n_jobs=20, n_estimators=100).fit(x_train_25, y_train_25)
    data2020_pred_pm25_1 = pred_metrics(linear_reg_pm25, pm_x_test_25, pm25_y_test, pmdata2020_25)
    data2020_pred_pm25_2 = pred_metrics(lasso_reg_pm25, pm_x_test_25, pm25_y_test, data2020_pred_pm25_1)
    data2020_pred_pm25_4 = pred_metrics(knn_reg_pm25, pm_x_test_25, pm25_y_test, data2020_pred_pm25_2)
    data2020_pred_pm25_5 = pred_metrics(tree_reg_pm25, pm_x_test_25, pm25_y_test, data2020_pred_pm25_4)
    data2020_pred_pm25_6 = pred_metrics(rf_reg_pm25, pm_x_test_25, pm25_y_test, data2020_pred_pm25_5)
    data2020_pred_pm25_7 = pred_metrics(ada_reg_pm25, pm_x_test_25, pm25_y_test, data2020_pred_pm25_6)
    data2020_pred_pm25_8 = pred_metrics(gbr_reg_pm25, pm_x_test_25, pm25_y_test, data2020_pred_pm25_7)
    data2020_pred_pm25_9 = pred_metrics(lgb_reg_pm25, pm_x_test_25, pm25_y_test, data2020_pred_pm25_8)
    # data2020_pred_pm25_9.columns = ['Date/time', 'station', 'PM2.5 [ug/m3]', 'PM10 [ug/m3]',
    #                                 'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
    #                                 'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
    #                                 'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
    #                                 'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday',
    #                                 'linear_pred','linear_resid','lasso_pred','lasso_resid',
    #                                 'knn_pred','knn_resid','tree_pred','tree_resid',
    #                                 'rf_pred','rf_resid','ada_pred','ada_resid','gbr_pred','gbr_resid',
    #                                 'lgb_pred','lgb_resid']
    # data2020_pred_pm25_9.to_csv('./out/csv_file/pmdata2020_pred_test_pm25.csv', index=False)
    estimators_pm25 = [("knn", knn_reg_pm25),("rf", rf_reg_pm25),("lgb", lgb_reg_pm25),("gbr", gbr_reg_pm25)]
    final_estimator_pm25 = MLPRegressor(hidden_layer_sizes=(5,5), activation='relu', 
                                solver='adam', random_state=42, max_iter=500)

    stack_reg_pm25 = StackingRegressor(estimators=estimators_pm25, final_estimator=final_estimator_pm25, cv=5)
    stack_fit_pm25  = stack_reg_pm25.fit(x_train_25, y_train_25)
    data2020_pred_pm25_10 = pred_metrics(stack_fit_pm25, x_test_25, y_test_25, data2020_pred_pm25_9)
    data2020_pred_pm25_10.columns = ['Date/time', 'station', 'PM2.5 [ug/m3]', 'PM10 [ug/m3]',
                                     'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                                     'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                                     'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                                     'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday',
                                     'linear_pred','linear_resid','lasso_pred','lasso_resid',
                                     'knn_pred','knn_resid','tree_pred','tree_resid',
                                     'rf_pred','rf_resid','ada_pred','ada_resid','gbr_pred','gbr_resid',
                                     'lgb_pred','lgb_resid', 'stack_pred','stack_resid']
    data2020_pred_pm25_10.to_csv('./out/csv_file/pmdata2020_pred_pm25_stack.csv', index=False)
    # dump(stack_fit_pm25, '.model/stack_trainedModel_pm25.joblib')

    x_train_10 = pm_x_train_10
    y_train_10 = pm10_y_train
    x_test_10 = pm_x_test_10
    y_test_10 = pm10_y_test
    linear_reg_pm10 = LinearRegression().fit(x_train_10, y_train_10)
    lasso_reg_pm10 = Lasso(alpha=0.1).fit(x_train_10, y_train_10)
    knn_reg_pm10 = neighbors.KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=20).fit(x_train_10, y_train_10)
    tree_reg_pm10 = DecisionTreeRegressor(random_state=42, max_depth=60, criterion='squared_error').fit(x_train_10, y_train_10)
    rf_reg_pm10 = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=5, n_jobs=20).fit(x_train_10, y_train_10)
    ada_reg_pm10 = AdaBoostRegressor(random_state=42, n_estimators=10, loss='square',learning_rate=0.01).fit(x_train_10, y_train_10)
    gbr_reg_pm10 = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=10, learning_rate=0.05, max_features='sqrt').fit(x_train_10, y_train_10)
    lgb_reg_pm10 = lgb.LGBMRegressor(random_state=42, n_jobs=20, n_estimators=50).fit(x_train_10, y_train_10)
    data2020_pred_pm10_1 = pred_metrics(linear_reg_pm10, x_test_10, y_test_10, pmdata2020_10)
    data2020_pred_pm10_2 = pred_metrics(lasso_reg_pm10, x_test_10, y_test_10, data2020_pred_pm10_1)
    data2020_pred_pm10_4 = pred_metrics(knn_reg_pm10, x_test_10, y_test_10, data2020_pred_pm10_2)
    data2020_pred_pm10_5 = pred_metrics(tree_reg_pm10, x_test_10, y_test_10, data2020_pred_pm10_4)
    data2020_pred_pm10_6 = pred_metrics(rf_reg_pm10, x_test_10, y_test_10, data2020_pred_pm10_5)
    data2020_pred_pm10_7 = pred_metrics(ada_reg_pm10, x_test_10, y_test_10, data2020_pred_pm10_6)
    data2020_pred_pm10_8 = pred_metrics(gbr_reg_pm10, x_test_10, y_test_10, data2020_pred_pm10_7)
    data2020_pred_pm10_9 = pred_metrics(lgb_reg_pm10, x_test_10, y_test_10, data2020_pred_pm10_8)
    # data2020_pred_pm10_9.columns = ['Date/time', 'station', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]',
    #                           'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
    #                           'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
    #                           'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
    #                           'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday',
    #                           'linear_pred','linear_resid','lasso_pred','lasso_resid',
    #                           'knn_pred','knn_resid','tree_pred','tree_resid',
    #                           'rf_pred','rf_resid','ada_pred','ada_resid','gbr_pred','gbr_resid',
    #                           'lgb_pred','lgb_resid']
    # data2020_pred_pm10_9.to_csv('./out/csv_file/pmdata2020_pred_test_pm10.csv', index=False)
    estimators_pm10 = [("lgb", lgb_reg_pm10),("gbr", gbr_reg_pm10)]
    final_estimator_pm10 = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', 
                               solver='adam', random_state=42, max_iter=500)
    stack_reg_pm10 = StackingRegressor(estimators=estimators_pm10, final_estimator=final_estimator_pm10, cv=5)
    stack_fit_pm10  = stack_reg_pm10.fit(x_train_10, y_train_10)
    data2020_pred_pm10_10 = pred_metrics(stack_fit_pm10, x_test_10, y_test_10, data2020_pred_pm10_9)
    data2020_pred_pm10_10.columns = ['Date/time', 'station', 'PM10 [ug/m3]', 'PM2.5 [ug/m3]',
                                     'NOX [ug/m3 eq. NO2]', 'NO2/NOX ratio',
                                     'O3 [ug/m3]', 'Radiation[W/m2] meteo', 'Temperature meteo',
                                     'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
                                     'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday',
                                     'linear_pred','linear_resid','lasso_pred','lasso_resid',
                                     'knn_pred','knn_resid','tree_pred','tree_resid',
                                     'rf_pred','rf_resid','ada_pred','ada_resid','gbr_pred','gbr_resid',
                                     'lgb_pred','lgb_resid', 'stack_pred','stack_resid']
    data2020_pred_pm10_10.to_csv('./out/csv_file/pmdata2020_pred_pm10_stack.csv', index=False)
    # dump(stack_fit_pm10, '.model/stack_trainedModel_pm10.joblib')

def main():
    workstation_Flag = False
    train_model(workstation_Flag)

if __name__ == "__main__":
    main()