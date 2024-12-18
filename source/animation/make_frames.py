import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.measure import label
import json


plt.rcParams.update(json.load(open('/projekt_agmwend/home_rad/Joshua/MasterArbeit/plot_settings.json')))
fs = 8
plt.rcParams['font.size'] = fs
plt.rcParams['axes.titlesize'] = fs
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['legend.fontsize'] = fs
plt.rcParams['figure.titlesize'] = fs


def rand_cmap(nlabels, type='soft', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def plot_frame(ds, x_slice, outpath, T_MIN, T_MAX, S_MIN, S_MAX, cmap):

    ds_sel = ds.sel(x=x_slice)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    caxs = []

    for ax in axs:
        divider = make_axes_locatable(ax)
        caxs.append(divider.append_axes('right', size='5%', pad=0.05))

    # Plot the temperature

    im_temp = ds_sel['skin_t'].plot(ax=axs[0], cmap='inferno', add_colorbar=False, vmin=T_MIN, vmax=T_MAX)
    cbar = fig.colorbar(im_temp, cax=caxs[0], orientation='vertical')       
    cbar.set_label(r'$T_\mathrm{skin}$ (°C)')

    im_pred = ds_sel['smoothed_prediction'].plot(ax=axs[1], cmap='Dark2', add_colorbar=False, levels=[.5, 1.5, 2.5, 3.5, 4.5])
    cbar = fig.colorbar(im_pred, cax=caxs[1], orientation='vertical')
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(['Open\nWater', 'Ice Water\nMix', 'Snow\nCovered', 'Thin\nIce'])

    im_seg = ds_sel['segmentation'].plot(ax=axs[2], cmap=cmap, add_colorbar=False, vmin=S_MIN, vmax=S_MAX)
    cbar = fig.colorbar(im_seg, cax=caxs[2], orientation='vertical')

    x0 = ds_sel.x.min().round(-1).values
    x1 = ds_sel.x.max().round(-1).values
    xticks = np.arange(x0, x1, 100) 

    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xticks(xticks)  
        ax.set_xticklabels(xticks/100) 

        ax.set_yticks([117, 217, 317, 417, 517])
        ax.set_yticklabels([r'-2km', r'-1km', r'0km', r'1km', r'2km'])
        ax.set_xlabel('Along track (km)')
        ax.set_ylabel('Across track (km)')
    plt.savefig(outpath, dpi=300)
    plt.close()

# Make the frames

def make_frames(inpath, FPS=60):
    ds = xr.open_dataset(inpath)
    T_MIN = ds.skin_t.min().values
    T_MAX = ds.skin_t.max().values
    S_MAX = ds.segmentation.max().values
    S_MIN = ds.segmentation.min().values
    #S_FINAL_MAX = ds.skimage_seg.max().values
    #S_FINAL_MIN = ds.skimage_seg.max().values
    tasks = []

    cmap = rand_cmap(len(np.unique(ds.segmentation.values)), type='bright', first_color_black=False, last_color_black=False, verbose=False)
    inpath_stump = inpath.split('/')[-1].split('_full.nc')[0]
    cmap2 = rand_cmap(len(np.unique(ds.skimage_seg.values)), type='bright', first_color_black=False, last_color_black=False, verbose=False)

    
    if not os.path.isdir(f'../../plots/ani/{inpath_stump}_v2'):
        os.makedirs(f'../../plots/ani/{inpath_stump}_v2')

    for i, x in tqdm(enumerate(ds.x.values[:-2000][::FPS]), total=len(ds.x.values[::FPS])):
        outpath = f'../../plots/ani/{inpath_stump}/frame_{i:04d}.png'
        x_slice = slice(x, x + 2000)
        tasks.append((ds, x_slice, outpath, T_MIN, T_MAX, S_MIN, S_MAX, cmap))
        #plot_frame(ds, x_slice, outpath, T_MIN, T_MAX, S_MIN, S_MAX)
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(plot_frame, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks)):
            _ = future.result()
    return f'../../plots/ani/{inpath_stump}'


def plot_segments_frame(ds, x_slice, outpath, T_MIN, T_MAX, S_MIN, S_MAX, S_FINAL_MIN, S_FINAL_MAX, cmap1, cmap2, debug=False):
    ds_sel = ds.sel(x=x_slice)

    fig, axs = plt.subplots(5, 1, figsize=(6.5, 6.5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    caxs = [make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05) for ax in axs]

    # Temperature plot
    im_temp = ds_sel['BT_5'].plot(ax=axs[0], cmap='inferno', add_colorbar=False, vmin=T_MIN, vmax=T_MAX)
    cbar_t = fig.colorbar(im_temp, cax=caxs[0], orientation='vertical', label=r'$T_\mathrm{S}$ (°C)', ticks=[-30, -25, -20, -15, -10, -5, 0])

    # Old prediction contour plot
    im_pred_old = ds_sel['old_prediction'].plot(ax=axs[1], cmap='Dark2', add_colorbar=False, levels=[0.5, 1.5, 2.5, 3.5, 4.5])
    cbar_old = fig.colorbar(im_pred_old, cax=caxs[1], orientation='vertical')
    cbar_old.set_ticks([1, 2, 3, 4])
    cbar_old.set_ticklabels(['OW', 'IWM', 'SC', 'TI'])

    # Segmentation with random colors
    im_seg = ds_sel['segmentation'].plot(ax=axs[2], cmap=cmap1, add_colorbar=False, vmin=S_MIN, vmax=S_MAX)
    fig.colorbar(im_seg, cax=caxs[2], orientation='vertical')

    # Smoothed prediction contour plot
    im_smooth = ds_sel['smoothed_prediction'].plot.contourf(ax=axs[3], cmap='Dark2', levels=[0.5, 1.5, 2.5, 3.5, 4.5], add_colorbar=False)
    cbar_smooth = fig.colorbar(im_smooth, cax=caxs[3], orientation='vertical')
    cbar_smooth.set_ticks([1, 2, 3, 4])
    cbar_smooth.set_ticklabels(['OW', 'IWM', 'SC', 'TI'])

    # Skimage segmentation plot
    im_skimage = ds_sel['skimage_seg'].plot(ax=axs[4], cmap=cmap2, add_colorbar=False, vmin=S_FINAL_MIN, vmax=S_FINAL_MAX)
    fig.colorbar(im_skimage, cax=caxs[4], orientation='vertical')

    # Set labels and ticks
    x_ticks = np.arange(x_slice.start, x_slice.stop, 500)
    x_ticklabels = (x_ticks - x_slice.start) / 100
    for ax in axs:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)
        ax.set_yticks([306 - 200, 306, 306 + 200])
        ax.set_yticklabels([-2, 0, 2])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('')
        ax.set_ylabel('')

    for cbar in [cbar_t, cbar_old, cbar_smooth]:
        cbar.ax.set_frame_on(False)
        cbar.ax.tick_params(which='both', length=0)
        cbar.ax.grid(False)

    axs[-1].set_xlabel('Along track (km)')
    axs[2].set_ylabel('Across track (km)')
    if debug == False:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()

