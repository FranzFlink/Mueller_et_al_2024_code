
import xarray as xr
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../lib')
from helper import timing_wrapper, apply_roughness, calc_local_roughness
from tqdm import tqdm
from dask.distributed import Client, LocalCluster  
import os
import warnings 
warnings.filterwarnings("ignore")
from xhistogram.xarray import histogram

#predictor = TabularPredictor.load("AutogluonModels/ag-20240521_080247")
predictor = TabularPredictor.load("../model/AutogluonModel_random_forest_VELOX")


ds = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-04_v_0.1.nc')


#@timing_wrapper
def prepare_load_predict(ds):
    ds_copy = ds.copy()
    timestep = ds.time
    date = timestep.dt.strftime('%Y-%m-%d')

    X1 = ds.isel(band=0)['BT_2D']
    X2 = ds.isel(band=1)['BT_2D'] - ds.isel(band=3)['BT_2D']
    X3 = ds.isel(band=2)['BT_2D'] - ds.isel(band=3)['BT_2D']
    X4 = ds.isel(band=3)['BT_2D'] - ds.isel(band=4)['BT_2D']
    X5 = ds.isel(band=2)['BT_2D'] - ds.isel(band=4)['BT_2D']
    X6 = ds['sur_rgh']
    X7 = ds['neighbor_mean']
    X8 = ds['neighbor_std']

    skt = ds.isel(band=3)['BT_2D'].values * 0.96 + 4.4


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
    ### this has to be further discussed: should we lift the temperature of potential open-water? 
    ### Right now, this decision seems to be arbitrary, but could be argued on the basis of emissivity (?)
    # ds_predicted['skt'] = xr.where(ds_predicted['label'] == 1, ds.isel(band=3)['BT_2D'].values + 4.4, ds_predicted['skin_t'])

    cold_water = (ds_predicted['label'] == 1) & (ds_predicted['skin_t'] < -2.5)
    water_ice_mix = water_ice_mix | cold_water
    ds_predicted['label'] = ds_predicted['label'].where(~water_ice_mix, 2)

    quality_flag = ds.BT_2D.isel(band=0).isnull().sum(dim=('x', 'y')) == 0 
    quality_flag = quality_flag.values

    ### Here, we calculate the skin temperature, which is just a linear function of the brightness temperature of the 11.7µm channel. The coefficients are from the simulation of the radiative transfer model
    ds_copy['skt'] = ds_copy['BT_2D'].isel(band=3)

    ds_copy['type_frac'] = histogram(ds_predicted['label'],bins=np.array([.5, 1.5, 2.5, 3.5, 4.5]), dim=['y', 'x']) / (635 * 512)
    ds_copy['SIC_upper'] = ds_copy['type_frac'].isel(label_bin=[1, 2, 3]).sum('label_bin')
    ds_copy['SIC_lower'] = ds_copy['type_frac'].isel(label_bin=[2, 3]).sum('label_bin')
    ds_copy['pred_proba'] = ds_predicted['pred_proba']
    ds_copy['label'] = ds_predicted['label']
    ds_copy['quality_flag'] = quality_flag
    ds_copy['skt'] = ds_copy['skt']
    ds_copy['lat'] = ds_copy['lat'].isel(band=3)
    ds_copy['lon'] = ds_copy['lon'].isel(band=3)
    ds_copy['alt'] = ds_copy['alt'].isel(band=3)

    return ds_copy


df_select = pd.read_csv('../../data/flight_segments_HALO.csv', index_col=0)

#ds_RF11 = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-30_v_0.1.zarr', chunks='auto')
#ds_RF10 = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-03-29_v_0.1.zarr', chunks='auto')
ds_RF12 = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/2022-04-01_v_0.1.zarr', chunks='auto')

# ci00 = df_select.loc['HALO-AC3_HALO_RF10_ci01']
# ci01 = df_select.loc['HALO-AC3_HALO_RF11_ci01']
# ci02 = df_select.loc['HALO-AC3_HALO_RF11_ci02']
# ci03 = df_select.loc['HALO-AC3_HALO_RF11_ci03']
# hl01 = df_select.loc['HALO-AC3_HALO_RF12_hl01']
hl02 = df_select.loc['HALO-AC3_HALO_RF12_hl02']
hl03 = df_select.loc['HALO-AC3_HALO_RF12_hl03']
hl04 = df_select.loc['HALO-AC3_HALO_RF12_hl04']
hl05 = df_select.loc['HALO-AC3_HALO_RF12_hl05']
hl06 = df_select.loc['HALO-AC3_HALO_RF12_hl06']
hl07 = df_select.loc['HALO-AC3_HALO_RF12_hl07']
hl08 = df_select.loc['HALO-AC3_HALO_RF12_hl08']
hl09 = df_select.loc['HALO-AC3_HALO_RF12_hl09']
hl10 = df_select.loc['HALO-AC3_HALO_RF12_hl10']

ds_RF12_hl02 = ds_RF12.sel(time=slice(hl02['start'], hl02['end']))
ds_RF12_hl03 = ds_RF12.sel(time=slice(hl03['start'], hl03['end']))
ds_RF12_hl04 = ds_RF12.sel(time=slice(hl04['start'], hl04['end']))
ds_RF12_hl05 = ds_RF12.sel(time=slice(hl05['start'], hl05['end']))
ds_RF12_hl06 = ds_RF12.sel(time=slice(hl06['start'], hl06['end']))
ds_RF12_hl07 = ds_RF12.sel(time=slice(hl07['start'], hl07['end']))
ds_RF12_hl08 = ds_RF12.sel(time=slice(hl08['start'], hl08['end']))
ds_RF12_hl09 = ds_RF12.sel(time=slice(hl09['start'], hl09['end']))
ds_RF12_hl10 = ds_RF12.sel(time=slice(hl10['start'], hl10['end']))

ds_list = [ds_RF12_hl02, ds_RF12_hl03, ds_RF12_hl04, ds_RF12_hl05, ds_RF12_hl06, ds_RF12_hl07, ds_RF12_hl08, ds_RF12_hl09, ds_RF12_hl10]
ds_names = ['RF12_hl02', 'RF12_hl03', 'RF12_hl04', 'RF12_hl05', 'RF12_hl06', 'RF12_hl07', 'RF12_hl08', 'RF12_hl09', 'RF12_hl10']

if __name__ == '__main__':


    cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='16GB', n_workers=10, threads_per_worker=1, dashboard_address=':21197')
    client = Client(cluster)


    for i, ds in enumerate(ds_list):

        ds = ds.load()
        ds = apply_roughness(ds)

        date = ds.time.dt.strftime('%Y-%m-%d').values[0]
        path_stump = f'/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/data/predicted/circles/{ds_names[i]}'
        if not os.path.exists(path_stump):
            os.makedirs(path_stump)

        for time in tqdm(ds['time']):
            str_time = time.dt.strftime('%Y-%m-%dT%H_%M_%S').values
            ### check if the file already exists
            # if os.path.exists(f'{path_stump}/predicted_{time}.nc'):
            #     continue
            ds_timestep = ds.sel(time=time)
            #ds_timestep = apply_roughness(ds_timestep)
            ds_predicted = prepare_load_predict(ds_timestep)
            ds_predicted.to_netcdf(f'{path_stump}/predicted_{str_time}.nc', mode='w')