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

<<<<<<< HEAD
=======

>>>>>>> bf634795b1d166d93bd7d1b80d6893db1f790007
}



@timing_wrapper
<<<<<<< HEAD
def main(date):
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--date', type=str, help='date to process')
    # args = argparser.parse_args()

    #date = args.date

    cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='16GB', n_workers=10, threads_per_worker=1, dashboard_address=':21197')
    client = Client(cluster)

    filenames = pd.read_csv('../../data/base_filenames.csv')
=======
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--date', type=str, help='date to process')
    args = argparser.parse_args()

    date = args.date

    cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='16GB', n_workers=20, threads_per_worker=1, dashboard_address=':21197')
    client = Client(cluster)

    filenames = pd.read_csv('~/MasterArbeit/filenames.csv')
>>>>>>> bf634795b1d166d93bd7d1b80d6893db1f790007

    filename = filenames['VELOX_seaice'].where(filenames['date'] == date).dropna().values[0]

    if filename != '-':

        ds = xr.open_zarr(filename)
        ds = ds.chunk({'time': 50, 'x': -1, 'y': -1, 'band': -1})#.persist()
        ds_new = apply_roughness(ds)
        ds.close()
        gc.collect()
        #ds_new.to_zarr(filename+'_v_0.1', mode='w', consolidated=True)

<<<<<<< HEAD
        filename = filename.replace('.zarr', '_v_0.2.nc')
=======
        filename = filename.replace('.zarr', '_v_0.1.nc')
>>>>>>> bf634795b1d166d93bd7d1b80d6893db1f790007
    
        ds_new.to_netcdf(filename, mode='w', engine='h5netcdf')
    client.close()


<<<<<<< HEAD
if __name__ == '__main__':
    
    dates = ['2022-03-14', '2022-03-16', '2022-03-20', '2022-03-21','2022-03-28', '2022-03-29','2022-03-30', '2022-04-01', '2022-04-04', '2022-04-07', '2022-04-08', '2022-04-10', '2022-04-11', '2022-04-12']

    for date in dates:

        main(date)
        print(f'Finished processing {date}')
        print('-----------------------------------')
=======

if __name__ == '__main__':
    main()
>>>>>>> bf634795b1d166d93bd7d1b80d6893db1f790007
