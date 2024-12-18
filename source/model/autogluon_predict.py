
import xarray as xr
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from source.lib.helper import timing_wrapper, apply_roughness, calc_local_roughness
from tqdm import tqdm
from dask.distributed import Client, LocalCluster  
import os

#predictor = TabularPredictor.load("AutogluonModels/ag-20240521_080247")
predictor = TabularPredictor.load("AutogluonModels/ag-20240702_134602")



ds = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-04_v_0.1.nc')
correction_fields = xr.open_dataset('../../data/correction_fields_v1.nc')



#@timing_wrapper
def prepare_load_predict(ds, correction_fields=correction_fields):

    timestep = ds.time
    date = timestep.dt.strftime('%Y-%m-%d')
    # select the correction field for the specific date
    cf = correction_fields.BT_2D.sel(time=date)

    bt = ds.BT_2D.values

    # correct the BT_2D
    bt_corrected = bt - cf

    # replace the BT_2D with the corrected values

    ds['BT_2D'].loc[{'band' : [1, 2, 3, 4]}]  = bt_corrected.isel(band=[1, 2, 3, 4])

    X1 = ds.isel(band=0)['BT_2D']
    X2 = ds.isel(band=1)['BT_2D'] - ds.isel(band=3)['BT_2D']
    X3 = ds.isel(band=2)['BT_2D'] - ds.isel(band=3)['BT_2D']
    X4 = ds.isel(band=3)['BT_2D'] - ds.isel(band=4)['BT_2D']
    X5 = ds.isel(band=2)['BT_2D'] - ds.isel(band=4)['BT_2D']
    X6 = ds['sur_rgh']
    X7 = ds['neighbor_mean']
    X8 = ds['neighbor_std']

    skt = ds.isel(band=3)['BT_2D'].values * 1.007 + 1.098

    df_X = pd.DataFrame({
        'BT_1' : X1.values.flatten(),
        'BT_25' : X2.values.flatten(),
        'BT_35' : X3.values.flatten(),
        'BT_56' : X4.values.flatten(),
        'BT_36' : X5.values.flatten(),
        'sur_rgh' : X6.values.flatten(),
        'neighbor_mean' : X7.values.flatten(),
        'neighbor_std' : X8.values.flatten()
    })

    pred_proba = predictor.predict_proba(df_X, model= 'RandomForestEntr').values.reshape(635, 507, 3)
    #pred_proba = predictor.predict_proba(df_X).values.reshape(635, 507, 3)

    ds_predicted = xr.Dataset(
        {
            'pred_proba': (('x', 'y', 'cl'), pred_proba),
            'label' : (('x', 'y'), np.argmax(pred_proba, axis=-1) * 2 + 1),
            'skin_t' : (('x', 'y'), skt),
            'BT_1' : X1,
        },
        coords={
            'x': ds.x,
            'y': ds.y,
            'cl': ['1', '3', '4']    
        }
    )

    ds_predicted['label'] = xr.where(ds_predicted['label'] == 5, 4, ds_predicted['label'])

    water_ice_mix = (ds_predicted['pred_proba'].sel(cl='1') > .25 ) & (ds_predicted['pred_proba'].sel(cl='3') > .25)

    ### add a second condition for the water_ice_mix: if the skin temperature is below -2.5°C; this is only applied to open water 

    cold_water = (ds_predicted['label'] == 1) & (ds_predicted['skin_t'] < -2.5)
    water_ice_mix = water_ice_mix | cold_water
    ds_predicted['label'] = ds_predicted['label'].where(~water_ice_mix, 2)

    #ice_snow_mix = (ds_predicted['pred_proba'].sel(cl='3') > .45 ) & (ds_predicted['pred_proba'].sel(cl='5') > .45)
    #ds_predicted['label'] = ds_predicted['label'].where(~ice_snow_mix, 4)

    return ds_predicted


filelist = [    
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-04_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-14_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-16_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-20_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-21_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-28_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-29_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-30_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-01_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-07_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-08_v_0.1.zarr',
    '/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-10_v_0.1.zarr',
]

if __name__ == '__main__':

    timeseries = pd.read_csv('../../data/velox_timesteps_in_miz_v1.csv')
    timeseries['time'] = pd.to_datetime(timeseries['time'])


    cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='16GB', n_workers=10, threads_per_worker=1, dashboard_address=':21197')
    client = Client(cluster)


    for file in filelist:

        ds = xr.open_dataset(file)
        #ds = ds.chunk({'time': 50, 'x': -1, 'y': -1, 'band': -1})#.persist()
        #ds = apply_roughness(ds)
        date = ds.time.dt.strftime('%Y-%m-%d').values[0]
        timeseries_date = timeseries.where(timeseries['time'].dt.strftime('%Y-%m-%d') == date).dropna()
        print(f'Processing {file} with {len(timeseries_date)} timesteps')

        for time in tqdm(timeseries_date['time']):
            time = time.strftime('%Y-%m-%dT%H:%M:%S')
            ### check if the file already exists

            if os.path.exists(f'../../data/predicted/v_0.5/predicted_{time}.nc'):
                continue
            try:
                ds_timestep = ds.sel(time=time)
                ds_timestep = apply_roughness(ds_timestep)
                ds_predicted = prepare_load_predict(ds_timestep)
                ds_predicted.to_netcdf(f'../../data/predicted/v_0.5/predicted_{time}.nc')
            except:
                print(f'Error processing {time}')
                continue