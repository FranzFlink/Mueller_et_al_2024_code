import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from tqdm import tqdm
from helper import calc_local_roughness, timing_wrapper
from dask_ml.wrappers import ParallelPostFit, Incremental

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from scipy.ndimage import gaussian_filter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV, RandomizedSearchCV

from dask.distributed import Client, LocalCluster
import dask.array as da

import dask
import argparse



@timing_wrapper
def train_random_forest(X, y, n_jobs=1, sigma=2.5, X_rgh=None):

    # Reshape the data to 2D
    X = X.reshape(-1, 5)
    y = y.reshape(-1)

    print('Inital unique labels:', np.unique(y))
    
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    if X_rgh is not None:
        r1, r2, r3 = X_rgh
        r1 = r1.flatten()
        r2 = r2.flatten()
        r3 = r3.flatten()

    mask = (y !=0 ) & ~np.isnan(y) & np.all(np.isfinite(X), axis=1) & (y != 3)

    # Remove pixels with label 0
    X_filtered = X[mask, :] 
    y_filtered = y[mask]
    r1 = r1[mask]
    r2 = r2[mask]
    r3 = r3[mask]

    dx1 = X_filtered[:, 1] - X_filtered[:, 4]
    dx2 = X_filtered[:, 2] - X_filtered[:, 4]
    dx3 = X_filtered[:, 3] - X_filtered[:, 4]

    X_filtered = np.stack((X_filtered[:, 0], dx1, dx2, dx3, r1, r2, r3), axis=1)

    #X_filtered = np.stack((X_filtered[:, 0], r2, r3), axis=1)


    print('Unique labels:', np.unique(y_filtered))

    N = len(y_filtered)

    print(len(y_filtered[y_filtered==1]) / N, 'Open Water')
    print(len(y_filtered[y_filtered==2]) / N, 'Dark Nilas')
    print(len(y_filtered[y_filtered==3]) / N, 'Bare Sea Ice')
    print(len(y_filtered[y_filtered==4]) / N, 'Snow-covered Sea Ice')


    y_filtered[y_filtered==1] = 1
    y_filtered[y_filtered==2] = 2
    y_filtered[y_filtered==3] = 3
    y_filtered[y_filtered==4] = 4


    model = RandomForestClassifier(random_state=42, n_jobs=n_jobs,
            n_estimators=50, max_depth=50, min_samples_split=2, min_samples_leaf=1, bootstrap=True,
            criterion='gini', max_features='sqrt',)
    
    #logo = LeaveOneGroupOut()

    scoring = 'accuracy'
    model = ParallelPostFit(model)
    #model = Incremental(model, scoring=scoring)

    with joblib.parallel_backend('dask'):
        model.fit(X_filtered, y_filtered)
    
    return model, #y_pred, metrics


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
    X1 = data['BT_2D'].isel(band=1) - data['BT_2D'].isel(band=4)
    X2 = data['BT_2D'].isel(band=2) - data['BT_2D'].isel(band=4)
    X3 = data['BT_2D'].isel(band=3) - data['BT_2D'].isel(band=4)
    X4 = data['sur_rgh']
    X5 = data['neighbor_mean']
    X6 = data['neighbor_std']

    XX = da.stack((X0, X1, X2, X3, X4, X5, X6), axis=0)
    #XX = da.stack((X0, X4, X5), axis=0)

    X_shape = XX.shape  
    target_shape = (data.time.size, data.x.size, data.y.size)

    print(X_shape, target_shape)    

    XX = XX.reshape(7, -1).T
    #pred = XX.map_partitions(
    #    dask_predict,
    #    model=clf_fut,
    #    meta=np.array([1]),
    #)

    da.rechunk(XX, chunks={0 : 100_000_000, 1 : 7})
    #pred = clf[0].predict(XX)


    scattered_model = dask.delayed(clf[0])


    preds = XX.map_blocks(
        dask_predict,
        model=scattered_model,
        dtype="int",
        drop_axis=1,
        meta=np.array([1]),
    )

    pred_2d = preds.reshape(target_shape)


    pred_2d = xr.Dataset(
        data_vars = {
            'label' : (['time', 'x', 'y'], pred_2d),
            'T_skin' : (['time', 'x', 'y'], data['BT_2D'].isel(band=4).values * 1.01 + 1.4)
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

    pred_2d.to_netcdf(f'/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/{date}_v_0.1_pred.nc', mode='w', engine='h5netcdf')




if __name__ == '__main__':

    cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='32GB', n_workers=16, threads_per_worker=1, dashboard_address=':8002')
    client = Client(cluster)


    #dates_to_precess = ['2022-03-28_v_0.1.nc'] processed @ 2024-02-11 10:00
    #dates_to_precess = ['2022-04-01_v_0.1.nc'] processed @ 2024-02-11 11:00
    #dates_to_precess = ['2022-04-04', '2022-04-07', '2022-04-08'] processed 2024-02-11 13:00 
    dates_to_process = ['2022-03-16', '2022-03-20', '2022-03-30']

    training_set = xr.open_zarr('/home/jomueller/MasterArbeit/ml_classification/test_ds.zarr')
    training_set = calc_local_roughness(training_set)

    X, y = training_set['BT_2D'].values, training_set['label'].values
    X_spatial = training_set['sur_rgh'].values, training_set['neighbor_mean'].values, training_set['neighbor_std'].values

    clf = train_random_forest(X, y, n_jobs=1, X_rgh=X_spatial)



    for date in dates_to_process:

        ### print some preambel
        print(f'Processing {date} ...')

        predict_parallel(date, clf)
