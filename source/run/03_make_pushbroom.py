import xarray as xr 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import os

def pixel_to_meter(pitch, roll, height, alpha=35.5, beta=28.7):

    pitch = np.radians(pitch)
    roll = np.radians(roll)
    alpha = np.radians(alpha)
    beta = np.radians(beta)

    xlen = (np.tan(alpha/2 + roll) + np.tan(alpha/2 - roll)) * height
    ylen = (np.tan(beta/2 + pitch) + np.tan(beta/2 - pitch)) * height

    return xlen, ylen


def concatenate_images(dataset, slicing_position=100, time_slice=None, channel=0, variable='BT_2D', quality_flag=None):

    df_filenames = pd.read_csv('/projekt_agmwend/home_rad/Joshua/MasterArbeit/filenames.csv')
    date = dataset.time.dt.strftime('%Y-%m-%d').values[0]
    bahamas_path = df_filenames['BAHAMAS'].where(df_filenames['date'] == date).dropna().values[0]
    ds_bahamas = xr.open_dataset(bahamas_path)

    xrHALO = xr.Dataset(
        data_vars=dict(
            lat=(["time"], ds_bahamas['IRS_LAT'].values),
            lon=(["time"], ds_bahamas['IRS_LON'].values),
            alt=(["time"], ds_bahamas['IRS_ALT'].values),
            roll=(["time"], ds_bahamas['IRS_PHI'].values),
            pitch=(["time"], ds_bahamas['IRS_THE'].values),
            hdg=(["time"], ds_bahamas['IRS_HDG'].values),
            gs=(["time"], ds_bahamas['IRS_GS'].values),
        ),
        coords=dict(

            time=ds_bahamas['TIME'].values,
        ),
    )

    if time_slice is not None:
        left_buffer = time_slice[0] 
        right_buffer = time_slice[-1]
        time_slice = slice(left_buffer, right_buffer)
    else:
        time_slice = slice(None)
    if variable == 'BT_2D':
        dataset_array = dataset[variable].isel(band=channel).sel(time=time_slice).to_numpy()
    else: 
        dataset_array = dataset[variable].sel(time=time_slice).to_numpy()
    dataset_time = dataset.time.sel(time=time_slice)
    xrHALO = xrHALO.resample(time='1s').nearest(tolerance='1s')
    xrHALO = xrHALO.sel(time=dataset_time)

    pixel_size_along_track = np.round(pixel_to_meter(xrHALO['pitch'], xrHALO['roll'], xrHALO['alt'])[1] / 507)
    pixel_size_across_track = np.round(pixel_to_meter(xrHALO['pitch'], xrHALO['roll'], xrHALO['alt'])[0] / 635)
    ground_speed = np.round(xrHALO['gs'])

    arrays_to_concat = []
    pixel_per_second = np.array(np.round(ground_speed / pixel_size_along_track, 0), dtype='int32')



    for i in range(len(dataset_array)):
        concating_array = dataset_array[i, :, slicing_position:slicing_position + pixel_per_second[i]]
        
        ### check if the slice has only constant values

        if quality_flag is not None:
            if quality_flag[i] == False:
                concating_array = np.zeros_like(concating_array) * np.nan
        arrays_to_concat.append(concating_array)

    im = np.concatenate(arrays_to_concat, axis=1)

    return im



def concatenate_images_delayed(ds_sel, slicing_position, variable, quality_flag, pad_size):
    #@delayed
    def concatenate_and_pad():
        im = concatenate_images(ds_sel, slicing_position=slicing_position, variable=variable, quality_flag=quality_flag)
        im = im.astype(np.float32)
        im = np.pad(im, ((0, 0), (slicing_position, 0)), mode='constant', constant_values=np.nan)
        im = np.pad(im, ((0, 0), (0, pad_size)), mode='constant', constant_values=np.nan)
        return im
    
    return concatenate_and_pad()

def average_concat_dask(ds_sel, variable='label', slicing_positions=np.arange(0, 500, 30)):
    quality_flag = ds_sel['quality_flag'].values
    results = [concatenate_images_delayed(ds_sel, sp, variable, quality_flag, slicing_positions[-(i+1)])
             for i, sp in enumerate(slicing_positions)]
    # with ProgressBar():
    #     results = compute(*tasks)
    

    ### select only results with the same shape

    results = [r for r in results if r.shape[1] == results[0].shape[1]]

    ds_cc = da.stack(results, axis=0)

    print(f'Calculating median of {variable} with shape {ds_cc.shape}')
    
    return ds_cc[8, :,:].compute()


def main(raw_file, processed_file, client):

    slicing_position = np.arange(0, 500, 30)

    outfile = os.path.basename(raw_file).replace('.nc', '_concat.nc')

    ds_raw = xr.open_dataset(raw_file, engine='h5netcdf').compute()
    ds_raw['quality_flag'] = ds_raw.BT_1.isnull().sum(dim=('x', 'y')) == 0 

    ds_processed = xr.open_dataset(processed_file, engine='h5netcdf')
    ds_processed['quality_flag'] = ds_processed.skin_t.isnull().sum(dim=('x', 'y')) == 0

    date = ds_raw.time.dt.strftime('%Y-%m-%d').values[0]
    ds_raw_bt5 = xr.open_dataset(f'/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/{date}_v_0.1.zarr', engine='zarr')['BT_2D'].isel(band=3).sel(time=ds_processed.time)
    ds_raw_tskin = ds_raw_bt5.compute()
    ds_raw_tskin = ds_raw_bt5.rename('skin_t')
    ds_raw_tskin['quality_flag'] = ds_raw_tskin.isnull().sum(dim=('x', 'y')) == 0

    print('Concating')
    cl_0 = average_concat_dask(ds_raw.isel(cl=0), variable='pred_proba', slicing_positions=slicing_position)
    cl_1 = average_concat_dask(ds_raw.isel(cl=1), variable='pred_proba', slicing_positions=slicing_position)
    cl_2 = average_concat_dask(ds_raw.isel(cl=2), variable='pred_proba', slicing_positions=slicing_position)
    bt_1 = average_concat_dask(ds_raw, variable='BT_1', slicing_positions=slicing_position)
    skin_t = average_concat_dask(ds_raw_tskin.to_dataset(), variable='skin_t', slicing_positions=slicing_position)
    lat = average_concat_dask(ds_processed, variable='lats', slicing_positions=slicing_position)
    lon = average_concat_dask(ds_processed, variable='lons', slicing_positions=slicing_position)

    ds_concat = xr.Dataset(
        data_vars=dict(
            cl_0=(["y", "x"], cl_0),
            cl_1=(["y", "x"], cl_1),
            cl_2=(["y", "x"], cl_2),
            bt_1=(["y", "x"], bt_1),
            skin_t=(["y", "x"], skin_t),
            lat=(["y", "x"], lat),
            lon=(["y", "x"], lon),
        ),
        coords=dict(
            x=np.arange(cl_0.shape[1]),
            y=np.arange(cl_0.shape[0]),
        ),
    )
    ds_concat.to_netcdf(f'../../data/cluster/input_sam_nadir/{outfile}', mode='w', engine='netcdf4')

    print(f'File {outfile} processed.')


import os
from stable_concat_to_cluster import main
from dask.distributed import Client, LocalCluster


if __name__ == '__main__':

    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='30GB')
    client = Client(cluster)

    raw_path = '/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/data/predicted/pushbroom/v_0.2/'
    processed_path = '/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/data/predicted/pushbroom_to_publish/v_0.2/'

    raw_files = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.nc')]
    processed_files = [os.path.join(processed_path, f) for f in os.listdir(processed_path) if f.endswith('.nc')]

    raw_files.sort()
    processed_files.sort()

    display_files = [f.split('/')[-1] for f in raw_files]

    for raw_file, processed_file in zip(raw_files, processed_files):


        selected_files = [
            #'2022-04-01T10:21:00_2022-04-01T10:54:00_concat.nc',
            '2022-04-01T12:20:00_2022-04-01T12:51:00_concat.nc',
            #'2022-04-04T13:19:30_2022-04-04T13:40:00_concat.nc',
            #'2022-04-01T09:25:00_2022-04-01T09:32:30_concat.nc',
        ]




        # #check if file already exists
        outfile = os.path.basename(raw_file).replace('.nc', '_concat.nc')
        # if os.path.isfile(f'../../data/cluster/input_sam_nadir/{outfile}'):
        #     print(f'File {outfile} already exists')
            
        if outfile in selected_files:
            print(f'Processing file {outfile}')
            main(raw_file, processed_file, client)