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

#test_ds_v4 = test_ds_v4.stack(z=('x', 'y', 'time')).isel(band=0).dropna('z' , how='any')  



def compute_seg_stats(ds_subset, ds_segment, ds_label):

    segments = np.unique(ds_segment)

    out = np.zeros((len(segments), 5))

    for i in range(0, len(segments)):

        ds_subset_ma = np.where(ds_segment == segments[i], ds_subset, np.nan)
        ds_subset_label = np.where(ds_segment == segments[i], ds_label, np.nan)
        ds_subset_label = ds_subset_label[~np.isnan(ds_subset_label)]
        #ds_subset = ds_subset.where(ds_subset.segmentation == segment)
        mean = np.nanmean(ds_subset_ma)
        std = np.nanstd(ds_subset_ma)
        mini = np.nanmin(ds_subset_ma)
        maxi = np.nanmax(ds_subset_ma)
        label = np.argmax(np.unique(ds_subset_label, return_counts=True)[1])

        out[i, :] = [mean, std, mini, maxi, label]

    return out

def apply_segment_stats(ds):
    out = pd.DataFrame()

    for i, time in tqdm(enumerate(ds.time)):

        ds_subset = ds['BT_2D'].isel(time=i, band=0).values
        ds_segment = ds['segmentation'].isel(time=i).values
        ds_label = ds['label'].isel(time=i).values

        pd.DataFrame(compute_seg_stats(ds_subset, ds_segment, ds_label)).T

        out = pd.concat([out, pd.DataFrame(compute_seg_stats(ds_subset, ds_segment, ds_label))])
        
            

    out.columns = ['mean', 'std', 'min', 'max', 'label']

# Assuming you have test_ds_v4 already defined
df_train = apply_segment_stats(test_ds_v4)

df_train.to_csv('segment_stats.csv')