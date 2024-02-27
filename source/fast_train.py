import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
import fsspec
import numpy as np
import xarray as xr
from scipy.stats import entropy as scipy_entropy



if os.path.isfile('/home/jomueller/MasterArbeit/ml_classification/test_ds_v3.nc'):
    test_ds_v2 = xr.open_dataset('/home/jomueller/MasterArbeit/ml_classification/test_ds_v3.nc')
    test_ds_v4 = xr.open_zarr('/home/jomueller/MasterArbeit/ml_classification/test_ds.zarr')
test_ds_v4 = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/test_ds_v5.nc')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from scipy.ndimage import gaussian_filter
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_val_predict



from helper import watershed_transformation
from tqdm import tqdm

def segment_training_data(ds):
    
    segments_array = np.zeros(ds['BT_2D'].isel(band=0).shape)   

    #give the segments a unique id 


    for i, time in tqdm(enumerate(ds.time)):
        segments = watershed_transformation(ds['BT_2D'].isel(time=i, band=0).values, feature_separation=5)
        segments_array[i, :, :] = segments + segments_array.max()


    ds['segmentation'] = (('time', 'x', 'y'), segments_array)
    
    return ds

test_ds_v4 = segment_training_data(test_ds_v4)



def compute_seg_stats(ds_subset, ds_segment, segment):

    print(ds_subset, ds_segment, segment)
    ds_subset = np.where(ds_segment == segment, ds_subset, np.nan)
    #ds_subset = ds_subset.where(ds_subset.segmentation == segment)
    mean = np.nanmean(ds_subset)
    std = np.nanstd(ds_subset)
    mini = np.nanmin(ds_subset)
    maxi = np.nanmax(ds_subset)

    return mean, std, mini, maxi

def apply_segment_stats(ds):
    segments = np.unique(ds.segmentation.values)
    mean, std, entropy, mini, maxi = xr.apply_ufunc(
        compute_seg_stats,
        ds.BT_2D.isel(band=0),
        ds.segmentation,
        segments,
        input_core_dims=[['x', 'y'], ['x', 'y'], ['segment']],
        output_core_dims=[['segment']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float, float, float]
    )

    ds_out = xr.Dataset(
        data_vars={
            'mean': (('segment',), mean),
            'std': (('segment',), std),
            'entropy': (('segment',), entropy),
            'mini': (('segment',), mini),
            'maxi': (('segment',), maxi)
        },
        coords={'segment': segments}
    )

    return ds_out

# Assuming you have test_ds_v4 already defined
test_ds_v4 = apply_segment_stats(test_ds_v4)