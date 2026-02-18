# -*- coding: utf-8 -*-
"""
Script to extract great circle path out of HARMONIE model

.. module:: test_harm_1d

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2016-2020, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from griblib import HARMONIE
from griblib.utils.geo import get_great_circle_path
from griblib.io.ascii import write_profile
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import BORDERS

# *************************************************************************
# parameters
# *************************************************************************

fid_grib = 'cy36/20170529_06/HARM_N25_201705290600_00000_GB'
fid_profile = 'harm20170529-06.dat'
fid_provinces = '/Users/assink/infrasound/network/maps/NL/GMT/Provinciegrenzen_2018-shp/Provinciegrenzen_2018.shp'

# plotting parameters
temp_lim = {'min': -5.0, 'max': 5.0}
temp_lim = {'min': 10.0, 'max': 25.0}
wind_lim = {'min': 2.0, 'max': 20.0}
z_lim = {'min': 0.0, 'max': 15.0}

# interpolation point
lon = 4.0
lat = 52.0

# *************************************************************************
# make great circle path outline
# *************************************************************************

# parameters for great circle path
path_params = dict()
path_params['type'] = 'range-azimuth'
path_params['start'] = dict(lat=51.0, lon=0.0)
# path_params['end'] = dict(lat=54.0, lon=9.0)
path_params['range'] = 700.0e3
path_params['azimuth'] = 60.0
dr = 1.0e3

(lons, lats, azi, dist) = get_great_circle_path(path_params, dr)
path_params['end'] = dict(lat=lats[-1], lon=lons[-1])

# *************************************************************************
# code
# *************************************************************************

my_atmos = HARMONIE()
ds = my_atmos.read_grib(fid_grib)

t_fcst=(ds.time + ds.step).dt.strftime('%Y%m%d-%H').values

# compute pressure, geopotential and geometric altitudes on model levels
ds = my_atmos.grib_compute_pressure_altitude_density(ds)

# *************************************************************************
# select a point from where to get a profile
# *************************************************************************

dspt = ds.interp(longitude=lon, latitude=lat)
write_profile(dspt, fid_profile)

# *************************************************************************
# tweaking output
# *************************************************************************

# Temperature (in degrees Celcius)
temperature = ds['t'].squeeze() - 273.15
#temperature -= temperature.mean(dim='distance')
temperature.attrs= {'long_name': 'temperature',
                    'units' : 'deg C',
                    'standard_name': 'air_temperature'}

# Wind speed
windspeed = np.sqrt(ds.u**2+ds.v**2).squeeze()
windspeed = windspeed.where(windspeed > 0.95*wind_lim['min'])
windspeed.attrs= {'long_name': 'wind speed',
                    'units' : 'm/s',
                    'standard_name': 'wind_speed'}

# *************************************************************************
# plotting values on ground level
# *************************************************************************

level = 60
n_plots = 3

new_lon = ds.longitude
new_lat = ds.latitude

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True,  figsize=(12,6),
                       subplot_kw={'projection': ccrs.PlateCarree()})

fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.15, hspace=0.15)

temperature.sel(hybrid=level).plot(cmap='Spectral_r',
                                             ax=ax[0,0],
                                             vmin=temp_lim['min'],
                                             vmax=temp_lim['max'])

windspeed.sel(hybrid=level).plot(cmap='inferno_r', 
                      ax=ax[1,0],
                      vmin=wind_lim['min'],
                      vmax=wind_lim['max'])


ds['w'].sel(hybrid=level).plot(cmap='RdBu_r',
                               ax=ax[0,1],
                               vmin=-0.2,
                               vmax=0.2)

ds['pp'].sel(hybrid=level).plot(cmap='RdBu_r',
                                ax=ax[1,1],
                                vmin=-2,
                                vmax=2)
ax[0,1].set_ylabel('')
ax[0,1].set_xlabel('')
ax[1,1].set_ylabel('')
ax[0,0].set_xlabel('')

for i in range(0,2):
    for j in range(0,2):
        ax[i,j].set_title('')
        ax[i,j].set_extent((new_lon[0], new_lon[-1], new_lat[0], new_lat[-1]))
        # add Borders and so
        ax[i,j].coastlines(linewidth=.8, edgecolor='gray')
        ax[i,j].add_feature(BORDERS, linewidth=.8, edgecolor='black')
        # add Dutch provinces
        prov2018 = list(shpreader.Reader(fid_provinces).geometries())
        ax[i,j].add_geometries(prov2018, ccrs.PlateCarree(), linewidth=1.0,
                          edgecolor='darkgray', facecolor='gray', alpha=0.2)

        ax[i,j].plot(lons, lats, linestyle='dashed', transform=ccrs.PlateCarree())
        ax[i,j].scatter(path_params['start']['lon']%360,
                        path_params['start']['lat'],
                        c='red', edgecolor='black')
        ax[i,j].scatter(path_params['end']['lon']%360,
                        path_params['end']['lat'],
                        c='red', edgecolor='black')
        ax[i,j].scatter(lon%360, lat,
                        c='lightblue', edgecolor='black')

# *************************************************************************
# plot topography
# *************************************************************************

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines(linewidth=.8, edgecolor='gray')
ax.add_feature(BORDERS, linewidth=.8, edgecolor='black')
# add Dutch provinces
prov2018 = list(shpreader.Reader(fid_provinces).geometries())
ax.add_geometries(prov2018, ccrs.PlateCarree(), linewidth=1.0,
                  edgecolor='darkgray', facecolor='gray', alpha=0.2)

ds['z'].sel(hybrid=level).plot(cmap='terrain',
                               ax=ax,
                               vmin=-50, vmax=350, zorder=0)
ax.set_extent((new_lon[0], new_lon[-1], new_lat[0], new_lat[-1]))

ax.plot(lons, lats, linestyle='dashed', transform=ccrs.PlateCarree())
ax.scatter(path_params['start']['lon'],
                path_params['start']['lat'],
                c='red', edgecolor='black')
ax.scatter(path_params['end']['lon'],
                path_params['end']['lat'],
                c='red', edgecolor='black')
ax.set_title('')
plt.show()
