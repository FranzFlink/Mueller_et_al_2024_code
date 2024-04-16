import numpy as np
from scipy.ndimage import convolve
from dask import array as da 
import time


from skimage import img_as_ubyte, feature, filters
from skimage.segmentation import watershed
from scipy import ndimage
import warnings
import numpy as np

import matplotlib.pyplot as plt

def calc_local_roughness(ds):
    roughness_array = da.zeros(ds['BT_2D'].isel(band=0).shape)
    neighbor_mean_array = da.zeros(ds['BT_2D'].isel(band=0).shape)
    neighbor_std_array = da.zeros(ds['BT_2D'].isel(band=0).shape)

    kernel = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 0, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]])

    #for i, time in tqdm(enumerate(ds.time), total=len(ds.time)):
    for i, time in enumerate(ds.time[:1000]):
        current_data = ds['BT_2D'].isel(time=i, band=0).values

        grad = np.gradient(current_data)

        roughness = np.sqrt(grad[0]**2 + grad[1]**2)
        roughness_array[i, :, :] = roughness

        neighbor_sum = convolve(current_data, kernel, mode='constant')
        neighbor_mean = neighbor_sum / 25
        neighbor_mean_array[i, :, :] = neighbor_mean

        neighbor_squared_sum = convolve(current_data**2, kernel, mode='constant')
        neighbor_variance = (neighbor_squared_sum / 25) - neighbor_mean**2
        neighbor_std = np.sqrt(neighbor_variance)
        neighbor_std_array[i, :, :] = neighbor_std

    ds['sur_rgh'] = (('time','x', 'y'), roughness_array)
    ds['neighbor_mean'] = (('time', 'x', 'y'), neighbor_mean_array)
    ds['neighbor_std'] = (('time', 'x', 'y'), neighbor_std_array)

    return ds


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



# print the classification report in a pretty way


def print_classification_report(classification_report):
    """
    Print the classification report in a pretty way.
    
    Parameters:
        classification_report (dict): The classification report.
    """
    print(f"{'Label':<20} {'Precision':<20} {'Recall':<20} {'F1-score':<20}")
    for label, metrics in classification_report.items():
        if label == 'accuracy': 
            print('accuracy_score:' , metrics)
            print('\n')

        else:
            print(f"{label:<20} {metrics['precision']:<20} {metrics['recall']:<20} {metrics['f1-score']:<20}")
            print('-'*80)

    # plot the confusion matrix in a pretty way

def plot_confusion_matrix(confusion_matrix):

    plt.figure(figsize=(6, 4))
    plt.imshow(confusion_matrix, cmap='Reds', )
    plt.title('Confusion matrix')
    plt.colorbar(label='Total number of samples')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks([1, 2, 3], ['Open Water', 'Thin Ice', 'Snow-covered Sea Ice'], rotation=45)
    plt.yticks([1, 2, 3], ['Open Water', 'Thin Ice', 'Bare Sea Ice', 'Snow-covered Sea Ice'], rotation=45)

    # plot the percentage of the confusion matrix into the figure

    
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, f'{confusion_matrix[i, j]:.3f}',
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > .9 else "black")






# Adjust the build_gradient_2d function to normalize the gradient before converting to 8-bit++
def edge_detect_2d(image_data, gauss_sigma, low_threshold, high_threshold):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        edge_image = img_as_ubyte(feature.canny(image_data, sigma=gauss_sigma,
                                                low_threshold=low_threshold, high_threshold=high_threshold))
    return edge_image



def build_gradient_2d_normalized(image_data, gauss_sigma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smooth_im = ndimage.filters.gaussian_filter(image_data, sigma=gauss_sigma)
        grad_image_float = filters.scharr(smooth_im)
        # Normalize the gradient to be between 0 and 1
        grad_image_float = (grad_image_float - grad_image_float.min()) / (grad_image_float.max() - grad_image_float.min())
        grad_image = img_as_ubyte(grad_image_float)

    # Prevent the watersheds from 'leaking' along the sides of the image
    grad_image[:, 0] = grad_image[:, 1]
    grad_image[:, -1] = grad_image[:, -2]
    grad_image[0, :] = grad_image[1, :]
    grad_image[-1, :] = grad_image[-2, :]

    return grad_image



def watershed_transformation(image_data, low_threshold=.1, high_threshold=.5, gauss_sigma=.1, feature_separation=10):
    """
    Runs a watershed transform on the 2D image.
    """
    
    # Build a raster of detected edges to inform the creation of watershed seed points
    edge_image = edge_detect_2d(image_data, gauss_sigma, low_threshold, high_threshold)
    
    # Build a raster of image gradient that will be the base for watershed expansion.
    grad_image = build_gradient_2d_normalized(image_data, gauss_sigma)

    # Find local minimum values in the edge image by inverting edge_image and finding the local maximum values
    inv_edge = np.empty_like(edge_image, dtype=np.uint8)
    np.subtract(255, edge_image, out=inv_edge)

    # Distance to the nearest detected edge
    distance_image = ndimage.distance_transform_edt(inv_edge)

    # Local maximum distance
    local_max_coordinates = feature.peak_local_max(distance_image, min_distance=feature_separation,
                                       exclude_border=False, num_peaks_per_label=1)
    
    markers = np.zeros_like(image_data, dtype=int)
    for marker_num, coordinates in enumerate(local_max_coordinates, start=1):
        markers[coordinates[0], coordinates[1]] = marker_num
    # Build a watershed from the markers on top of the edge image
    im_watersheds = watershed(grad_image, markers)

    return im_watersheds