# -*- coding: utf-8 -*-
"""
Script to extract great circle path out of ECMWF model

.. module:: test_ecmwf_gcp_slice

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2016-2020, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from griblib import ECMWF
import numpy as np
import matplotlib.pyplot as plt

# *************************************************************************
# parameters
# *************************************************************************

fid_grib = 'analysis/AN_ECMWF_201705290600.grib'

path_params = dict()
path_params['type'] = 'range-azimuth'
path_params['start'] = dict(lat=51.0, lon=0.0)
#path_params['type'] = 'coordinates'
#path_params['end'] = dict(lat=54.0, lon=9.0)
path_params['range'] = 700.0e3
path_params['azimuth'] = 60.0

temp_lim = {'min': -5.0, 'max': 5.0}
#temp_lim = {'min': -20.0, 'max': 20.0}
wind_lim = {'min': 0.0, 'max': 30.0}
z_lim = {'min': 0.0, 'max': 15.0}
z_ticks = np.arange(z_lim['min'], z_lim['max'], 5.0)

dr = 1.0e3

# *************************************************************************
# code
# *************************************************************************

my_atmos = ECMWF()
ds = my_atmos.read_grib(fid_grib)

t_fcst=(ds.time + ds.step).dt.strftime('%Y%m%d-%H').values

# extract slice along great circle path
ds_gcp = my_atmos.extract_gcp_slice(ds, path_params, dr=dr)

# compute pressure, geopotential and geometric altitudes on model levels
ds_gcp = my_atmos.grib_compute_pressure_altitude_density(ds_gcp)

# *************************************************************************
# tweaking output
# *************************************************************************

# Temperature (in degrees Celcius)
temperature = ds_gcp['t'].squeeze() - 273.15
temperature -= temperature.mean(dim='distance')
temperature.attrs= {'long_name': 'temperature',
                    'units' : 'deg C',
                    'standard_name': 'air_temperature'}

# Wind speed
windspeed = np.sqrt(ds_gcp.u**2+ds_gcp.v**2).squeeze()
#windspeed = windspeed.where(windspeed > 0.95*wind_lim['min'])
windspeed.attrs= {'long_name': 'wind speed',
                    'units' : 'm/s',
                    'standard_name': 'wind_speed'}

# *************************************************************************
# plotting
# *************************************************************************

nlev = ds_gcp.dims['hybrid']
nrng = ds_gcp.dims['distance']

x = np.broadcast_to(np.expand_dims(ds_gcp.distance, axis=0), shape=(nlev, nrng)) / 1e3
y = ds_gcp.z / 1e3

n_plots = 3
fig, ax = plt.subplots(n_plots, 1, sharex=True, figsize=(8,5))
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.15, hspace=0.15)

im = ax[0].pcolormesh(x, y, temperature,
             cmap='Spectral_r', shading='gouraud',
             vmin=temp_lim['min'], vmax=temp_lim['max'])
plt.colorbar(im,ax=ax[0],label='$\Delta$T [deg C]')

im = ax[1].pcolormesh(x, y, windspeed,
             cmap='inferno_r', shading='gouraud',
             vmin=wind_lim['min'], vmax=wind_lim['max'])

plt.colorbar(im,ax=ax[1],label='|u| [m/s]')

im = ax[2].pcolormesh(x, y, ds_gcp.w,
             cmap='RdBu_r', shading='gouraud',
             vmin=-0.2, vmax=0.2)
plt.colorbar(im,ax=ax[2],label='w [m/s]')

for i in range(0,n_plots):
    ax[i].set_title('')
    ax[i].set_xlabel('')
    ax[i].set_ylabel('Altitude [km]')
    ax[i].set_ylim(z_lim['min'], z_lim['max'])
    ax[i].set_yticks(z_ticks)
    ax[i].grid()

lons = ds_gcp.longitude.values
lats = ds_gcp.latitude.values
coord_str = '({slo:.1f}E,{sla:.1f}N) > ({elo:.1f}E,{ela:.1f}N)'.format(
    slo=lons[0], sla=lats[0], elo=lons[-1], ela=lats[-1])
title_str = '{model} fcst {time} UT - gcp {coords}'.format(
    model=my_atmos.model, time=t_fcst, coords=coord_str)
ax[0].set_title(title_str)

ax[2].set_xlabel('Distance [km]')

plt.show()