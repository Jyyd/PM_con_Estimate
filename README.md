<!--
 * @Author: jyyd23@mails.tsinghua.edu.cn
 * @Date: 2024-05-25 00:49:48
 * @LastEditors: jyyd23@mails.tsinghua.edu.cn
 * @LastEditTime: 2024-05-25 01:12:43
 * @FilePath: \con_code\README.md
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
-->

# My_Project: PM_con_estimate

## 1.Description

**Innovative Stacking Method for Enhanced Data Fusion in Pollutant Population Risk Evaluation**
![weighted_pop_pm25](https://github.com/Jyyd/PM_con_Estimate/blob/main/figure/weighted_pop_pm25.png)
![weighted_pop_pm10](https://github.com/Jyyd/PM_con_Estimate/blob/main/figure/weighted_pop_pm10.png)
![swiss_pm10_pop](https://github.com/Jyyd/PM_con_Estimate/blob/main/figure/swiss_pm10_pop.png)
![swiss_pm25_pop](https://github.com/Jyyd/PM_con_Estimate/blob/main/figure/swiss_pm25_pop.png)


## 2.File Structure
```
con_code
├─ figure
├─ PM_DL.py
├─ PM_pop.py
├─ PM_pred.py
├─ PM_stack.py
├─ test_plot.ipynb
├─ tif_file
│  ├─ annual
│  └─ wash_tiff
├─ wash_data.py
└─ __pycache__
   └─ PM_pop.cpython-39.pyc

```
## 3.File Function

### 3.1 High-Resolution Particulate Matter Estimation via Trained Stacking 
1. wash_data.py: Get the feature of PM.
2. PM_stack.py: The Stacking train process.
3. PM_DL.py: Get Deep Learning results.
4. PM_pred.py: The pred process for high-resolution PM.

### 3.2 PM_estimate population and landcover
1. PM_pop.py: The code to work on population and land cover.
2. test_plot.ipynb : The code to plot figures.

## 4.Software and Hardware Environment and Version

### 4.1 Hardware

#### 4.1.1 Computer
+ **CPU**: Intel(R) Core(TM) i7-10750H
+ **GPU**: NVIDIA Geforce GTX 1650 Ti
+ **RAM**: 16GB

#### 4.1.2 Workstation
+ **CPU**: Intel(R) Xeon(R) Platinum 8383C
+ **RAM**: 256GB

### 4.1 Software

#### 4.2.1 Computer
+ **Operating System**: Windows 11
+ **Python Version**: Anaconda Python 3.9.7
+ **CUDA Version**: 11.7

#### 4.2.2 Workstation
+ **Operating System**: Windows 10
+ **Python Version**: Anaconda Python 3.11.4

### 4.3 Environment configuration commands

 ```cmd
    pip install -r requirements.txt
```

# 5 Training Command

## 5.1 Wash Data
 ```cmd
    python wash_data.py
```

## 5.2 Training with Stacking model
 ```cmd
    python PM_stack.py
```

## 5.3 Pred high-resolution PM
 ```cmd
    python PM_pred.py
```

## 5.4 Result PM
1. ```cmd
   pip3 install runipy
   ```
2. ```cmd
   runipy test_plot.ipynb
   ```
