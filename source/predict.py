#### predict the surface mask for VELOX images 
#### using a pre-trained model;

import os
import sys
import joblib
import numpy as np
import xarray as xr
from tqdm import tqdm
from dask.distributed import Client, LocalCluster
import dask.array as da 


base_path = f"/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/ml_sea-ice-classification/models/rf"
model_path = os.path.join('/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/ml_sea-ice-classification/models', 'final_model_3_class_no_bare.joblib')


if __name__ == '__main__':





    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        clf = joblib.load(model_path)




    #cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='10GB', n_workers=32, threads_per_worker=1)
    #client = Client(cluster)    

    ds_pred = xr.open_zarr('/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-01.zarr_v_0.1/', chunks={'time': 100, 'x': 250, 'y': 250})    

    XX = ds_pred['BT_2D'].values.reshape(-1, 5)
    X_rgh = ds_pred['sur_rgh'].values, ds_pred['neighbor_mean'].values, ds_pred['neighbor_std'].values

    X = np.stack((XX[:, 0], XX[:, 1] - XX[:, 4], XX[:, 2] - XX[:, 4], XX[:, 3] - XX[:, 4], X_rgh[0].flatten(), X_rgh[1].flatten(), X_rgh[2].flatten()), axis=1)

    X = np.stack((XX[:,0], XX[:,1], XX[:,2], XX[:,3], XX[:,4], X_rgh[0].flatten(), X_rgh[1].flatten(), X_rgh[2].flatten()), axis=1)

    X = X.reshape(ds_pred.time.size, -1 , 8)
    shape_2d = ds_pred.BT_2D.isel(time=0, band=0).shape
    ds_pred['prediction_pixel'] = (('time', 'x', 'y'), da.zeros(ds_pred.BT_2D.isel(band=0).shape))


    for i, time in tqdm(enumerate(ds_pred.time)):

        x = X[i].reshape(-1, 8)

        mask = ~np.isnan(x).any(axis=1)

        x = x[mask, :]
        if len(x) != 0:
            y_pred = clf.predict(x)
        else: 
            y_pred = np.nan 


        y_pred_2D = np.nan * np.ones(shape_2d) 

        y_pred_2D = y_pred_2D.reshape(-1)
        y_pred_2D[mask] = y_pred
        y_pred_2D = y_pred_2D.reshape(shape_2d)

        ds_pred['surface_mask'].loc[time] = y_pred_2D


    ds_pred['skin_T'] = ds_pred['BT_2D'].isel(band=3) * 1.01 + 1.4 

    ds_out = ds_pred.drop(['BT_2D', 'sur_rgh', 'neighbor_mean', 'neighbor_std'])
    ds_out.to_zarr('/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-01.zarr_v_0.2/', mode='w')