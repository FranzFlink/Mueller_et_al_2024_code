from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xarray as xr
import numpy as np
from tqdm import tqdm
from flox.xarray import xarray_reduce
from helper import watershed_transformation
from tqdm import tqdm
from dask.distributed import Client, LocalCluster

def segment_training_data(ds):
    
    segments_array = np.zeros(ds['BT_2D'].isel(band=0).shape)   

    for i, time in tqdm(enumerate(ds.time)):
        segments = watershed_transformation(ds['BT_2D'].isel(time=i, band=0).values, feature_separation=5)
        segments_array[i, :, :] = segments + segments_array.max()

    ds['segmentation'] = (('time', 'x', 'y'), segments_array)
    
    return ds

if __name__ == '__main__':

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        memory_limit='8GB',
        dashboard_address=8000,
    )

    client = Client(cluster)

    train_ds = xr.open_dataset('training.nc', chunks={'time' : -1, 'x' : 250, 'y' : 250}, engine='h5netcdf')



    train_ds = segment_training_data(train_ds)

    train_ds_flat = train_ds.stack(feature=['x', 'y', 'time'])
    train_ds_flat = train_ds_flat.chunk({'feature': -1})

    means = xarray_reduce(train_ds_flat.BT_2D, train_ds_flat.segmentation, func='nanmean', engine='flox', expected_groups=np.unique(train_ds_flat.segmentation))
    var = xarray_reduce(train_ds_flat.BT_2D, train_ds_flat.segmentation, func='nanvar', engine='flox', expected_groups=np.unique(train_ds_flat.segmentation))
    maxs = xarray_reduce(train_ds_flat.BT_2D, train_ds_flat.segmentation, func='nanmax', engine='flox', expected_groups=np.unique(train_ds_flat.segmentation))
    mins = xarray_reduce(train_ds_flat.BT_2D, train_ds_flat.segmentation, func='nanmin', engine='flox', expected_groups=np.unique(train_ds_flat.segmentation))
    labels = xarray_reduce(train_ds_flat.label, train_ds_flat.segmentation, func='nanmode', engine='numpy', method='blockwise', expected_groups=np.unique(train_ds_flat.segmentation))

    ### write the training plaquette to disk

    train_plaquette = xr.Dataset({'mean': means, 'var': var, 'max': maxs, 'min': mins, 'label': labels})
    train_plaquette = train_plaquette.to_array().stack(feature=('variable', 'band')).dropna('feature', how='any')

    print('Lazy until here')

    train_plaquette.to_pandas().to_csv('training_segment.csv')


    


