import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from scipy.stats import linregress
from haversine import haversine, Unit
from sklearn.metrics import mean_squared_error
import json
import cartopy
import cartopy.crs as ccrs


MODIS = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/MODIS/MODIS_Aqua_Sea_Ice_Extent_and_IST_Daily_L3_Global_4km_EASE-Grid_Day_V061.nc')
VELOX = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/MasterArbeit/HALO-AC3_VELOX_coarsen_100_v1s.nc')
VELOX_new = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_unified_data/unified_velox_nadir_Filter_01.nc')
surface_mask = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_unified_data/unified_surface_mask.nc')
gps = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_unified_data/unified_gps_new.nc')


dates = np.unique(VELOX_new['time'].dt.date)

dates = [date.strftime('%Y-%m-%d') for date in dates]

cross_corr_dataset = []


for date in tqdm(dates):

    plt.rcParams.update(json.load(open("../plot_settings.json", "r")))
    subset = MODIS.sel(time=date)
    vel_sel = surface_mask.sel(time=date)
    gps_sel = gps.sel(time=date).interp_like(vel_sel, method='nearest')
    vel_sel[list(gps_sel.data_vars)] = gps_sel

    lons = vel_sel.lon.values
    lats = vel_sel.lat.values

    result_ds = xr.Dataset(
        data_vars=dict(
            MODIS_skin_T=(['time'], np.zeros(len(lons))),
            VELOX_skin_T=(['time'], np.zeros(len(lons))),
            distances=(['time'], np.zeros(len(lons))),
            sf_mask=(['time'], np.zeros(len(lons))),
            dist_mask=(['time'], np.zeros(len(lons))),
        ),
        coords=dict(
            time=vel_sel.time.values,
        )
    )


    for i, (lon, lat) in tqdm(enumerate(zip(lons, lats))):
        

        closest_lat = float(subset.lat.sel(lat=lat, method='nearest').values)
        closest_lon = float(subset.lon.sel(lon=lon, method='nearest').values)
        lon = lon.astype(float)
        lat = lat.astype(float)

        distance = haversine((lat, lon), (closest_lat, closest_lon), unit=Unit.KILOMETERS)
        mod_sst = subset.sel(lat=closest_lat, lon=closest_lon).IST.values.item()

        if (distance < 1) & ~np.isnan(mod_sst):

            result_ds['MODIS_skin_T'][i] = mod_sst
            result_ds['VELOX_skin_T'][i] = vel_sel.isel(time=i)['Vfull_mean'] + 273.15
            result_ds['distances'][i] = distance
            result_ds['sf_mask'][i] = vel_sel.sel(time=vel_sel.time.values[i]).mask.values
            result_ds['dist_mask'][i] = 1


    times = result_ds.time.where(result_ds.dist_mask == 1).values
    sf_mask = result_ds.sf_mask.where(result_ds.dist_mask == 1).values
    MODIS_skin_T = result_ds.MODIS_skin_T.where(result_ds.dist_mask == 1).values
    VELOX_skin_T = result_ds.VELOX_skin_T.where(result_ds.dist_mask == 1).values

    fig, ax = plt.subplot_mosaic(
        """
        AAFF
        AAFF
        BBBB
        CCCC
        DDEE
        DDEE
        """
        ,
        figsize=(13, 8),
        gridspec_kw={
            'hspace': 1,
            'wspace': 0.3
        },
        per_subplot_kw={'A': {'projection': ccrs.NorthPolarStereo()}}
    )


    cloud_mask = xr.where((sf_mask == 0) | (sf_mask ==2), True, False)



    ax['B'].plot(times ,MODIS_skin_T, color='red', lw=1, ls='-', marker='o', fillstyle='none', alpha=.2, markersize=2)
    ax['B'].plot(times ,VELOX_skin_T, color='blue', lw=1, ls='-', marker='o', fillstyle='none', alpha=.2, markersize=2)

    ax['B'].plot(times[cloud_mask] ,MODIS_skin_T[cloud_mask], color='red', label='MODIS', lw=1, ls='-', marker='o', markersize=2)
    ax['B'].plot(times[cloud_mask] ,VELOX_skin_T[cloud_mask], color='blue', label='VELOX', lw=1, ls='-', marker='o', alpha=.8, markersize=2)

    ax['B'].legend()
    ax['B'].set_xlabel('Time')
    ax['B'].set_ylabel('Skin Temperature [K]')
    ax['C'].plot(result_ds.distances)
    ax['C'].set_xlabel('Time')  
    ax['C'].set_ylabel('Distance from \nMODIS to VELOX [km]')

    subset.IST.plot(ax=ax['A'], transform=ccrs.PlateCarree(), cmap='plasma')


    ax['A'].add_feature(cartopy.feature.LAND, zorder=1, facecolor='gray')
    ax['A'].add_feature(cartopy.feature.COASTLINE, zorder=1, edgecolor='black')


    im = ax['A'].scatter(lons, lats, c=vel_sel.mask, s=2, cmap='viridis', transform=ccrs.PlateCarree())
    cb = fig.colorbar(im, ax=ax['A'])
    cb.set_ticks(ticks=[0, 1, 2, 3],
        labels=['clear & ocean', 'cloudy & ocean', 'clear & sea-ice', 'cloudy & sea-ice'])

    mask = ~np.isnan(MODIS_skin_T) & ~np.isnan(VELOX_skin_T) 

    Y = np.array(MODIS_skin_T)[mask]
    X = np.array(VELOX_skin_T)[mask]


    print(f'MSE: {mean_squared_error(X, Y):.2f}')

    ### make two fits, one over open water and one over ice


    for i, mask_index in enumerate([0, 2]):

        Y = np.array(MODIS_skin_T) - 273.15
        X = np.array(VELOX_skin_T) - 273.15
        sf_mask = np.array(sf_mask)

        nan_mask = ~np.isnan(X) & ~np.isnan(Y)

        mask = (sf_mask == mask_index) & nan_mask

        y = Y[mask]
        x = X[mask]

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        color = 'red' if mask_index == 0 else 'blue'

        i = 'D' if mask_index == 0 else 'E'

        ax[i].plot(x, slope*x + intercept, color='black', label=f'Linear Regression\nR^2: {r_value**2:.2f}', lw=1, ls='--')
        ax[i].scatter(x, y, color=color, s=10)
        mask_title = 'Open Water' if mask_index == 0 else 'Ice'
        ax[i].set_title(f'Mask: {mask_title}; R²: {r_value**2:.2f}\n; Slope: {slope:.2f}; Intercept: {intercept:.2f}')

    mask = ~np.isnan(MODIS_skin_T) & ~np.isnan(VELOX_skin_T)

    y = np.array(MODIS_skin_T)[mask] - 273.15
    x = np.array(VELOX_skin_T)[mask] - 273.15

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    ax['F'].plot(x, slope*x + intercept, color='blue', label=f'Linear Regression: ax+b\nR²: {r_value**2:.2f}\na : {slope:.2f}\nb: {intercept:.2f}', lw=1, ls='-')
    ax['F'].scatter(x, y, color='red', s=10)
    #print(f'R^2: {r_value**2:.2f}')
    #print(f'Slope: {slope}')
    #print(f'Intercept: {intercept}')

    ax['F'].set_xlabel(r'VELOX $T_{skin}$[K]')
    ax['F'].set_ylabel(r'MODIS $T_{skin}$[K]')

    ax['F'].plot([-28, 5], [-28, 5], color='black', ls='--', lw=1, label='1:1 Line')
    ax['F'].legend()
    
    plt.savefig(f'../../plots/sst_overview/{date}_comparison.png')


    cross_corr_dataset.append(result_ds)

cross_corr_dataset = xr.concat(cross_corr_dataset, dim='time')
cross_corr_dataset.to_netcdf('/projekt_agmwend/home_rad/Joshua/HALO-AC3_unified_data/unified_cross_corr_dataset.nc')