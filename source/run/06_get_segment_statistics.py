import os
import xarray as xr
import numpy as np
from tqdm import tqdm
from skimage.morphology import remove_small_objects
from skimage.measure import label as ski_label
from flox.xarray import xarray_reduce
from dask.distributed import Client, LocalCluster

input_path = '../../../data/cluster/input_sam_nadir'
output_path = '../../../data/cluster/output_sam'

infiles = sorted([f'{input_path}/{f}' for f in os.listdir(input_path)])
outfiles = sorted([f'{output_path}/{f}' for f in os.listdir(output_path) if 'T' in f and 'nc' not in f])

if __name__ == '__main__':

    for i in range(len(infiles)):
        key = infiles[i].split('/')[-1].split('_concat.nc')[0]
        if os.path.exists(infiles[i]) and os.path.exists(f'{output_path}/{key}'):
            print(f'{key} exists: {len(os.listdir(f"{output_path}/{key}") )} output-files')

        ds_in = xr.open_dataset(infiles[i], chunks='auto')
        ds_out = xr.open_dataset(f'{output_path}/{key}_full.nc', chunks='auto')
        ds_in['skin_t'] = ds_in['skin_t'] * .96 + .14
        seg_map = ds_out['segmentation'].values
        max_proba = ds_in[['cl_0', 'cl_1', 'cl_2']].interpolate_na(dim='x', method='linear').to_array()
        max_proba = max_proba.pad(variable=(0, 1), constant_values=0)
        max_proba['variable'] = np.array([0, 1, 2, 3])
        max_proba = max_proba.argmax('variable').values
        #max_proba = max_proba.where(max_proba != 3).values
        smoothed_prediction = np.zeros_like(max_proba)

        water_ice_mix = (ds_in['cl_0'] > .25 ) & (ds_in['cl_1'] > .25 )
        cold_water = (ds_out['smoothed_prediction'] == 1) & (ds_out['skin_t'] < -3)
        water_ice_mix = water_ice_mix | cold_water
        ds_out['smoothed_prediction'] = ds_out['smoothed_prediction'].where(~water_ice_mix, 2)

        for seg in tqdm(np.unique(seg_map)):
            uniques, counts = np.unique(max_proba[seg_map == seg], return_counts=True)
            most_occuring_label = uniques[np.argmax(counts)]
            smoothed_prediction[seg_map == seg] = most_occuring_label

        ds_out['smoothed_prediction'] = xr.DataArray(smoothed_prediction + 1, dims=['y', 'x'])
        ds_out['smoothed_prediction'] = xr.where(ds_out.smoothed_prediction == 2, 4, xr.where(ds_out.smoothed_prediction == 4, 2, ds_out.smoothed_prediction))

        image = ds_out['smoothed_prediction'].values
        label_image = ski_label(image, connectivity=2)

        ds_out['label'] = xr.DataArray(label_image, dims=['y', 'x'])
        ds_out['label'] = (('y', 'x'),remove_small_objects(ds_out['label'].values, min_size=1))


        # Apply Flox-based aggregation
        unique_ws = np.unique(ds_out['label'].values)

        segment_area = xarray_reduce(ds_out['label'], ds_out['label'], func='count', engine='flox', expected_groups=unique_ws)
        segment_T = xarray_reduce(ds_out['skin_t'], ds_out['label'], func='nanmean', engine='flox', expected_groups=unique_ws)
        segment_std = xarray_reduce(ds_out['skin_t'], ds_out['label'], func='nanstd', engine='flox', expected_groups=unique_ws)
        segment_label = xarray_reduce(ds_out['smoothed_prediction'], ds_out['label'], func='nanmode', engine='numpy', expected_groups=unique_ws)
        segment_lat = xarray_reduce(ds_out['lat'], ds_out['label'], func='nanmean', engine='flox', expected_groups=unique_ws)
        segment_lon = xarray_reduce(ds_out['lon'], ds_out['label'], func='nanmean', engine='flox', expected_groups=unique_ws)

        # Store results in dataset
        ds_segstats = xr.Dataset(
            data_vars={
                'segment_size': ('segment', segment_area.values),
                'segment_T': ('segment', segment_T.values),
                'segment_std': ('segment', segment_std.values),
                'segment_label': ('segment', segment_label.values),
                'segment_area': ('segment', segment_area.values),
                'segment_lat': ('segment', segment_lat.values),
                'segment_lon': ('segment', segment_lon.values),
            },
            coords={'segment': np.arange(len(segment_area))}
        )

        # Save results
        ds_segstats_path = f'/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/data/segmented_data_nadir/{key}.nc'
        if not os.path.isfile(ds_segstats_path):
            ds_segstats.to_netcdf(ds_segstats_path)
            print('Saved segmentation statistics')

        ds_full_path = f'/projekt_agmwend/home_rad/Joshua/Mueller_et_al_2024/data/cluster/output_sam/{key}_full.nc'
        if not os.path.isfile(ds_full_path):
            ds_out.to_netcdf(ds_full_path.replace('cluster/output_sam', 'full_datasets'))
            print('Saved full dataset')

