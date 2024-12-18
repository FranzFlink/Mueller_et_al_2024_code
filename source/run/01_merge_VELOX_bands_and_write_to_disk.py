import pandas as pd
from tqdm import tqdm
import argparse
import os
import xarray as xr
import numpy as np



#import warnings
#warnings.filterwarnings('ignore')


# A simple workflow to write all bands of a single date to disk 
# in a single zarr file. zarr is a format that is optimized for
# parallel read and write operations.


def write_all_channel_merged_to_disk(date, all_bands_filenames, cluster):

	client = cluster	

	# argparser = argparse.ArgumentParser()
	# argparser.add_argument('--date', type=str, help='date to process')
	# args = argparser.parse_args()
	#date = args.date

	list_of_datasets = []

	#for date in all_bands_filenames.columns:

	print(date)


	#look for the timeslices that are available for the date 
	#df_timeslice = currated_timeslices[currated_timeslices['date'] == date].dropna()


	for file in all_bands_filenames[date].sort_values():

		#print(file)	
		ds_xr = xr.open_dataset(file, chunks={'x': 250, 'y': 250, 'time': 250})
		ds_xr = ds_xr.drop_vars(['vza', 'vaa'])
		ds_concat = ds_xr
		#ds_concat = xr.concat([ds_xr.sel(time=eval(timeslice)) for timeslice in df_timeslice['slices']], dim='time')

		list_of_datasets.append(ds_concat)

	merged_dataset = xr.concat(list_of_datasets, dim='band')
	merged_dataset = merged_dataset.chunk({'x': 250, 'y': 250, 'time': 250, 'band': -1})

	out_file = f'/projekt_agmwend/home_rad/Joshua/HALO-AC3_VELOX_sea_ice/{date}_v_0.1.zarr'
	write_job = merged_dataset.to_zarr(out_file, mode='w', consolidated=True)

	### wait for the write job to finish, then close the client and return a signal that the job is done

	return write_job
			

import pandas as pd
from dask.distributed import Client, progress, LocalCluster
from dask.diagnostics import ProgressBar

if __name__ == '__main__':

    df = pd.read_csv('../../data/base_filenames.csv')
    dates = df.columns

    dates = ['2022-04-04', '2022-04-07', '2022-04-08', '2022-04-10', '2022-04-11', '2022-04-12']


    cluster = LocalCluster(local_directory='/projekt_agmwend/home_rad/Joshua/dask-worker-space', memory_limit='10GB', n_workers=32, threads_per_worker=1)
    client = Client(cluster)
    print(f'Visit me @ {client.dashboard_link}')


    for date in dates:
        print(f'Processing {date}')

        signal = write_all_channel_merged_to_disk(date, df, client)

        if signal:
            print(f'Finished processing {date}')



    
