from helper import apply_roughness, timing_wrapper
import pandas as pd
import argparse
import xarray as xr
from dask.distributed import Client, LocalCluster
import gc

encoding = {
    'BT_2D' : {'chunks' : (50, -1, -1, -1)},
    'sur_rgh' : {'chunks' : (50, -1, -1)},
    'neighbor_mean' : {'chunks' : (50, -1, -1)},
    'neighbor_std' : {'chunks' : (50, -1, -1)},
    'BT_Center' : {'chunks' : (50, -1)},
    ## i think these geve me the encoding headache before... ## 
    'time': {'chunks' : (50, -1)},
    'alt' : {'chunks' : (50, -1)},
    'lat' : {'chunks' : (50, -1)},
    'lon' : {'chunks' : (50, -1)},
    'yaw' : {'chunks' : (50, -1)},


}



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
        gc.collect()
        #ds_new.to_zarr(filename+'_v_0.1', mode='w', consolidated=True)

        filename = filename.replace('.zarr', '_v_0.1.nc')
    
        ds_new.to_netcdf(filename, mode='w', engine='h5netcdf')
    client.close()



if __name__ == '__main__':
    main()
