import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from tqdm import tqdm
from source.lib.helper import calc_local_roughness, timing_wrapper, print_classification_report
from dask_ml.wrappers import ParallelPostFit

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler


from dask.distributed import Client, LocalCluster
import dask.array as da

import dask
import argparse



@timing_wrapper
def train_random_forest(X, y, n_jobs=1, sigma=2.5, X_rgh=None):


    mask = np.isnan(X).any(axis=1) | np.isnan(y) 
    X = X[~mask]
    y = y[~mask]


    model = RandomForestClassifier(random_state=42, n_jobs=n_jobs,
            n_estimators=50, max_depth=50, min_samples_split=2, min_samples_leaf=1, bootstrap=True,
            criterion='gini', max_features='sqrt',)

    rus = RandomUnderSampler(random_state=0)

    X_resampled, y_resampled = rus.fit_resample(X, y)

    ### validate the model
    print('Cross validating ...')
    #y_pred = cross_val_predict(model, X_resampled, y_resampled, cv=5, n_jobs=40, verbose=10)

    #class_report = classification_report(y_resampled, y_pred, output_dict=True)
    #print_classification_report(class_report)

    model = ParallelPostFit(model)
    #model = Incremental(model, scoring=scoring)

    with joblib.parallel_backend('dask'):
        model.fit(X_resampled, y_resampled)
        #model.fit(X, y)
    return model #y_pred, metrics


def dask_predict(part, model):
    return model.predict(part)


@timing_wrapper
def predict_parallel(date, clf):
    #argparser = argparse.ArgumentParser()
    #argparser.add_argument('--date', type=str, help='date to process')
    #args = argparser.parse_args()

    #date = args.date

    #base_path = f"/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/ml_sea-ice-classification/models/rf"
    #model_path = os.path.join('/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/ml_sea-ice-classification/models', 'final_model_3_class_no_bare.joblib')

    # for now, we have to retrain every time, because the model is not pickable ... 
    #clf = train_random_forest(X, y, n_jobs=1, X_rgh=X_add)

    data = xr.open_dataset(f'/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/{date}_v_0.1.nc', chunks={'time' : 250, 'x' : 250, 'y' : 250, 'band' : 5})#.isel(time=slice(0, 1000)).persist()

    X0 = data['BT_2D'].isel(band=0)
    X1 = data['BT_2D'].isel(band=1)# - data['BT_2D'].isel(band=4)
    X2 = data['BT_2D'].isel(band=2)# - data['BT_2D'].isel(band=4)
    X3 = data['BT_2D'].isel(band=3)# - data['BT_2D'].isel(band=4)
    X4 = data['BT_2D'].isel(band=4)
    X5 = data['sur_rgh']
    X6 = data['neighbor_mean']
    X7 = data['neighbor_std']

    XX = da.stack((X0, X1, X2, X3, X4, X5, X6, X7), axis=0)
    #XX = da.stack((X0, X4, X5), axis=0)

    X_shape = XX.shape  
    print(f'Processing {date} with shape {X_shape} ...')
    target_shape = (data.time.size, data.x.size, data.y.size)


    XX = XX.reshape(8, data.time.size * data.x.size * data.y.size).T
    #pred = XX.map_partitions(
    #    dask_predict,
    #    model=clf_fut,
    #    meta=np.array([1]),
    #)
    print(f'Rechunking to {XX.shape} ...')
    da.rechunk(XX, chunks={0 : 100_000_000, 1 : 8})
    #pred = clf[0].predict(XX)


    print(f'Delaying model ... {clf}')
    scattered_model = dask.delayed(clf)


    preds = XX.map_blocks(
        dask_predict,
        model=scattered_model,
        dtype="int",
        drop_axis=1,
        meta=np.array([1]),
    )

    print(f'Shape of predictions: {preds.shape}')

    print(f'Reshaping to {target_shape} ...')
    pred_2d = preds.reshape(target_shape)


    pred_2d = xr.Dataset(
        data_vars = {
            'label' : (['time', 'x', 'y'], pred_2d),
            'T_skin' : (['time', 'x', 'y'], data['BT_2D'].isel(band=3).values * 1.01 + 1.4)
        },
        coords = {
            'time' : data.time,
            'x' : data.x,
            'y' : data.y
        },
    )

    pred_2d.attrs = data.attrs
    pred_2d.attrs['label'] = '1: Open Water, 2: Thin Ice, 3: Snow-covered Sea Ice'
    pred_2d.attrs['T_skin'] = 'Skin temperature in °C'

    #pred_2d = xr.DataArray(pred_2d, coords={'time': data.time, 'x': data.x, 'y': data.y}, dims=['time', 'x', 'y'])
    #pred_2d = xr.Dataset(data_vars={'label' : pred_2d})

    pred_2d.to_netcdf(f'/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/{date}_v_0.2_pred.nc', mode='w', engine='h5netcdf')


if __name__ == '__main__':

    cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='32GB', n_workers=16, threads_per_worker=1, dashboard_address=':8002')
    client = Client(cluster)


    #dates_to_precess = ['2022-03-28_v_0.1.nc'] processed @ 2024-02-11 10:00
    #dates_to_precess = ['2022-04-01_v_0.1.nc'] processed @ 2024-02-11 11:00
    #dates_to_precess = ['2022-04-04', '2022-04-07', '2022-04-08'] processed 2024-02-11 13:00 
    #dates_to_process = ['2022-03-16', '2022-03-20', '2022-03-30'] processed 2024-02-11 14:00; v.0.2 @ 2024-03-22 10:00

    dates_to_process = ['2022-04-04'] # 
    #dates_to_process = ['2022-03-16'] 

    training_set = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/data/training_flat_corrected.nc')

    X = training_set[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].to_array().values.T
    y = training_set['y_corrected'].values


    clf = train_random_forest(X, y, n_jobs=1)


    for date in dates_to_process:

        ### print some preambel
        print(f'Processing {date} ...')

        predict_parallel(date, clf)
