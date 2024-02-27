import numpy as np
from scipy.ndimage import convolve
from dask import array as da 
import time
import pandas as pd
import argparse
import xarray as xr
from dask.distributed import Client, LocalCluster
import gc


def calc_local_roughness_slice(bt_2d_slice):
    kernel = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 0, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]])

    grad = np.gradient(bt_2d_slice)

    roughness = np.sqrt(grad[0]**2 + grad[1]**2)

    neighbor_sum = convolve(bt_2d_slice, kernel, mode='constant')
    neighbor_mean = neighbor_sum / 25

    neighbor_squared_sum = convolve(bt_2d_slice**2, kernel, mode='constant')
    neighbor_variance = (neighbor_squared_sum / 25) - neighbor_mean**2
    neighbor_std = np.sqrt(neighbor_variance)

    return roughness, neighbor_mean, neighbor_std


import xarray as xr

# Assuming `ds` is your xarray.Dataset and 'BT_2D' is the variable you're working with
def apply_roughness(ds):
    roughness, neighbor_mean, neighbor_std = xr.apply_ufunc(
        calc_local_roughness_slice, 
        ds['BT_2D'].isel(band=0),  # Adjust this if your input data structure is different
        input_core_dims=[['x', 'y']], 
        output_core_dims=[['x', 'y'], ['x', 'y'], ['x', 'y']], 
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float]
    )

    # Assuming 'time', 'x', 'y' are the dimensions of your original data variable
    ds['sur_rgh'] = (('time', 'x', 'y'), roughness.values)
    ds['neighbor_mean'] = (('time', 'x', 'y'), neighbor_mean.values)
    ds['neighbor_std'] = (('time', 'x', 'y'), neighbor_std.values)

    return ds


### make a timing wrapper 


def timing_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Executed {func.__name__} in {end - start} seconds")
        return result
    return wrapper




@timing_wrapper
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, help='date to process')
    args = argparser.parse_args()

    date = args.date

    cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='16GB', n_workers=20, threads_per_worker=1, dashboard_address=':21197')
    client = Client(cluster)

    filenames = pd.read_csv('~/MasterArbeit/filenames.csv')

    filename = filenames['VELOX_seaice'].where(filenames['date'] == date).dropna().values[0]

    if filename != '-':

        ds = xr.open_zarr(filename)
        ds = ds.chunk({'time': 50, 'x': -1, 'y': -1, 'band': -1})#.persist()
        ds_new = apply_roughness(ds)
        ds.close()
        filename = filename.replace('.zarr', '_v_0.1.nc')
    
        ds_new.to_netcdf(filename, mode='w', engine='h5netcdf')
    client.close()



if __name__ == '__main__':
    main()
