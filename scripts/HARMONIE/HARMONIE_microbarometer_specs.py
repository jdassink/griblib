from griblib import HARMONIE
from griblib.utils.geo import get_great_circle_path
from griblib.io.ascii import write_profile
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import BORDERS

from scipy import ndimage
import pandas as pd
import glob

import xarray as xr

############## FUNCTIONS ######################

def read_HARMONIE_set_one(self, fid_grid):
    request = [
        {'level_type': 'hybrid', 'var_list': ['pdep', 'tke']},
        {'level_type': 'heightAboveSea'},
        {'level_type': 'heightAboveGround', 'var_list': ['ugst', 'vgst', 'mld']}
    ]

    if self.cycle == 36:
        request.append({'level_type': 'heightAboveGround',
                        'var_list': ['tp']})
    else:
        request.append({'level_type': 'heightAboveGround',
                        'var_list': ['rain'], 'step_type': 'instant' })

    ds = self.read_grib(fid_grib, request, compat='override')
    return ds


def read_HARMONIE_set_two(self, fid_grid):
    request = [
        {'level_type': 'heightAboveGround', 'var_list': ['pres', 't']}
    ]

    ds = self.read_grib(fid_grib, request, verbose=False, compat='override')
    return ds

def read_HARMONIE(self, fid_grib):
    # Read data
    ds = read_HARMONIE_set_one(self, fid_grib)
    ground_level = len(ds.hybrid)

    ds_sfc = read_HARMONIE_set_two(self, fid_grib)
    
    # Wind speed
    ds['windgust_speed'] = np.sqrt(ds.ugst**2+ds.vgst**2)
    #ds.windgusts = windgusts.where(windgusts > 0.95*wind_lim['min'])
    ds.windgust_speed.attrs= {'long_name': 'Wind gust speed',
                                 'units' : 'm/s',
                                 'standard_name': 'wind_gust'}

    # Mean sea level pressure
    ds['pres_msl'] = ds.pres / 1e2
    ds.pres_msl.attrs={'long_name': 'MSL Pressure',
                       'units' : 'hPa',
                       'standard_name': 'MSL Pressure'}

    # Pressure model processing
    pres_msl_deviations = ds.pres_msl - ds.pres_msl.mean()
    # pres_msl_deviations.attrs={'long_name': 'MSL Pres. deviation',
    #                     'units' : 'hPa',
    #                     'standard_name': 'MSL Pres. deviation'}

    lowpass = ndimage.gaussian_filter(pres_msl_deviations.values, 10)
    gauss_highpass = pres_msl_deviations.values - lowpass

    ds['pres_hp'] = ds.pres_msl.copy(data=gauss_highpass).rename('pres_hp')
    ds.pres_hp.attrs = {'long_name': 'HP filtered MSL Pressure',
                        'units' : 'hPa',
                        'standard_name': 'HP filtered MSL Pressure'}

    ds['pres_lp'] =ds.pres_msl.copy(data=lowpass).rename('pres_lp')
    ds.pres_lp.attrs = {'long_name': 'LP filtered MSL Pressure',
                        'units' : 'hPa',
                        'standard_name': 'LP filtered MSL Pressure'}

    # Turbulent (pressure)
    t_ground = ds_sfc.t
    p_ground = ds_sfc.pres
    ds_sfc['den'] = self.compute_density(p_ground, t_ground)
    den_ground = ds_sfc.den.sel(heightAboveGround=0)

    # Turbulent pressure
    tke_ground = ds.tke.sel(hybrid=ground_level)
    ds['pres_turb'] = (den_ground * tke_ground).rename('pres_turb')
    #pressure_turbulence
    ds.pres_turb.attrs={'long_name': 'Turbulent pressure',
                        'units' : 'Pa',
                        'standard_name': 'Turbulent pressure'}
    ds['tke_ground'] = tke_ground
    ds.tke_ground.attrs = ds.tke.attrs    
    
    # Rain (convert kg/m^2/s = mm/s to mm/h)
    if self.cycle == 36:
        ds['rain'] = ds.tp*3600
    else:
        ds['rain'] = ds['rain'].sel(heightAboveGround=0)*3600

    ds.rain.attrs={'long_name': 'Rain',
                   'units' : 'mm/h',
                   'standard_name': 'Rain'}


    # Mixed layer height
    if self.cycle == 36:
        ds['mld'] = ds.p3067
        
    return ds

def sample_model(ds, df):
    ds_samples = []

    pts_lat = xr.DataArray(df.Latitude, dims='site_index')
    pts_lon = xr.DataArray(df.Longitude, dims='site_index')

    ds.load()
    ds_samples = ds.interp(longitude=pts_lon, latitude=pts_lat)

    ds_samples = ds_samples.drop(['pdep', 'tke', 'hybrid', 'heightAboveGround'])
    ds_samples = ds_samples.assign_coords(site_code=("site_index", df.Site.values))
    ds_samples = ds_samples.assign_coords(site_location=("site_index", df.Location.values))
    return ds_samples

############# /END FUNCTIONS #######################

############# PLOTTING FUNCTIONS #####################

def plot_prefix(self, ds):
    t_fcst=(ds.valid_time).dt.strftime('%Y%m%d-%H').values

    title_str = '{model}{cycle} forecast {time} UT'.format(
        model=self.model, cycle=self.cycle, time=t_fcst)

    fid_prefix = '{model}{cycle}_{time}'.format(model=self.model, 
                                                cycle=self.cycle, 
                                                time=t_fcst)
    return (t_fcst, title_str, fid_prefix)

def plot_pressure(self, ds):
    (t_fcst, title_str, fid_prefix) = plot_prefix(self, ds)
    ground_level = len(ds.hybrid)
    
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,6),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.01, hspace=0.15)

    ds.pres_msl.plot(cmap='RdBu_r',
                  ax=ax[0,0],
                  vmin=P_lim['min'], vmax=P_lim['max'],
                  robust=True,
                  transform=ccrs.PlateCarree())
    levels = np.arange(980,1030,2)
    ds.pres_msl.plot.contour(ax=ax[0,0],
                           colors='black',
                           levels=levels,
                           alpha=0.2,
                           linewidths=2.0, 
                           transform=ccrs.PlateCarree())
    ds.pres_msl.plot.contour(ax=ax[0,0],
                           colors='black',
                           levels=levels,
                           alpha=0.9,
                           linewidths=0.25, 
                           transform=ccrs.PlateCarree())

    ###################

    ds.pdep.sel(hybrid=ground_level).plot(cmap='RdBu_r',
                  ax=ax[0,1],
                  vmin=-10, vmax=10,
                  robust=True,
    #               vmin=-1, vmax=1,
                  transform=ccrs.PlateCarree())
    # msl_filt.plot.contour(ax=ax[0],
    #                       colors='black',
    #                       levels=levels,
    #                       linewidths=.25, 
    #                       transform=ccrs.PlateCarree())

    ###################
    ds.pres_lp.plot(cmap='RdBu_r',
                    ax=ax[1,0],
                    robust=True,
                    vmin=-3, vmax=3,
                    transform=ccrs.PlateCarree())
    levels = np.arange(-4,4,1)
    ds.pres_lp.plot.contour(ax=ax[1,0],
                            colors='black',
                            levels=levels,
                            linewidths=.25, 
                            transform=ccrs.PlateCarree())
    ###################
    ds.pres_hp.plot(cmap='RdBu_r',
                    ax=ax[1,1],
                    robust=True,
                    vmin=-1, vmax=1,
                    transform=ccrs.PlateCarree())
    levels = np.arange(-2,2,1)
    # msl_filt.plot.contour(ax=ax[0],
    #                       colors='black',
    #                       levels=levels,
    #                       linewidths=.25, 
    #                       transform=ccrs.PlateCarree())

    for i in range(0,2):
        for j in range(0,2):
            ax[i,j].set_title('')
            ax[i,j].set_extent((ds.longitude.min(), ds.longitude.max(),
                                ds.latitude.min(), ds.latitude.max()))
            # add Borders and so
            ax[i,j].coastlines(linewidth=.8, edgecolor='gray')
            ax[i,j].add_feature(BORDERS, linewidth=.8, edgecolor='black')
            # add Dutch provinces
            ax[i,j].add_geometries(prov2018, ccrs.PlateCarree(), linewidth=1.0,
                              edgecolor='darkgray', facecolor='gray', alpha=0.2)

            #Plot great-circle path
            ax[i,j].scatter(path_params['start']['lon']%360,
                            path_params['start']['lat'],
                            c='C0', edgecolors='white', marker='o',
                            transform=ccrs.PlateCarree())
            ax[i,j].scatter(path_params['end']['lon']%360,
                            path_params['end']['lat'],
                            c='C0', edgecolors='white', marker='o',
                            transform=ccrs.PlateCarree())
            ax[i,j].plot(lons, lats, linestyle='dashed', transform=ccrs.PlateCarree(),
                         path_effects=[pe.Stroke(linewidth=5, foreground='w', alpha=0.75), pe.Normal()])
            # Plot (random) sampling point
            ax[i,j].scatter(df.Longitude, df.Latitude,
                            c='white', edgecolor='black', transform=ccrs.PlateCarree())
            
    fig.suptitle(title_str)

    fid = '{}_map_pressure.png'.format(fid_prefix)
    fig.savefig(fid, facecolor='white')
    #fig.savefig('../'+fid, facecolor='white')

def plot_meteo(self, ds):
    (t_fcst, title_str, fid_prefix) = plot_prefix(self, ds)
    ground_level = len(ds.hybrid)

    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True,  figsize=(12,10),
                           subplot_kw={'projection': ccrs.PlateCarree()})

    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, wspace=0.1, hspace=0.1)


    ####### MEAN SEA LEVEL PRESSURE ##############
    ds.pres_msl.plot(cmap='RdBu_r',
                      ax=ax[0,0],
                      robust=True,
                      vmin=P_lim['min'], vmax=P_lim['max'],
                      extend='both',
                      transform=ccrs.PlateCarree())
    levels = np.arange(980,1030,2)
    ds.pres_msl.plot.contour(ax=ax[0,0],
                           colors='black',
                           levels=levels,
                           alpha=0.2,
                           linewidths=2.0, 
                           transform=ccrs.PlateCarree())
    ds.pres_msl.plot.contour(ax=ax[0,0],
                           colors='black',
                           levels=levels,
                           alpha=0.9,
                           linewidths=0.25, 
                           transform=ccrs.PlateCarree())

    ####### WINDSPEED and DIRECTION ##############
    ds.windgust_speed.plot(cmap='inferno_r',
                           ax=ax[1,0],
                           robust=True,
                           vmin=wind_lim['min'], vmax=wind_lim['max'],
                           extend='both',
                           transform=ccrs.PlateCarree())

    # reinterpolate for wind arrows
    _lon = np.arange(ds.longitude.min(), ds.longitude.max(), 0.5)
    _lat = np.arange(ds.latitude.min(), ds.latitude.max(), 0.5)
    ds.load()
    u_ = ds.ugst.interp(latitude=_lat, longitude=_lon)
    v_ = ds.vgst.interp(latitude=_lat, longitude=_lon)
    w_ = np.sqrt(u_**2+v_**2)
    u_ = u_.where(w_ > 0.95*wind_lim['min']) / w_
    v_ = v_.where(w_ > 0.95*wind_lim['min']) / w_

    ax[1,0].quiver(_lon, _lat, u_.values, v_.values,
                    scale=25,
                    edgecolor='w',
                    width=0.005,
                    linewidth=0.5,
                    transform=ccrs.PlateCarree())

    ####### PRESSURE DEPARTURE ########################
    # ds['pdep'].sel(hybrid=ground_level).plot(cmap='RdBu_r',
    #                                 ax=ax[0,1],
    #                                 robust=True,
    #                                 vmin=-10, vmax=10,
    #                                 transform=ccrs.PlateCarree())
    ds.pres_hp.plot(cmap='RdBu_r',
                    ax=ax[0,1],
                    robust=True,
                    vmin=-1, vmax=1,
                    extend='both',
                    transform=ccrs.PlateCarree())


    ####### TURBULENT PRESSURE ##############
    # turbulence = ds['tke'].sel(hybrid=ground_level)
    # turbulence.attrs={'long_name': 'Turbulent Kinetic Energy',
    #             'units' : 'm$^2$/s$^2$',
    #             'standard_name': 'TKE'}
    ds.pres_turb.plot(cmap='gist_stern_r',
                        ax=ax[1,1],
                        robust=True,
                        vmin=ptke_lim['min'],vmax=ptke_lim['max'],
                        extend='both',
                        transform=ccrs.PlateCarree())
    ds.pres_msl.plot.contour(ax=ax[1,1],
                                colors='black',
                                levels=levels,
                                alpha=0.2,
                                linewidths=2.0, 
                                transform=ccrs.PlateCarree())
    ds.pres_msl.plot.contour(ax=ax[1,1],
                                colors='black',
                                levels=levels,
                                alpha=0.9,
                                linewidths=0.25, 
                                transform=ccrs.PlateCarree())

    ####### PRECIPITATION AND ISOBARS ##############

    ds.rain.plot(cmap='gist_stern_r',
                    ax=ax[2,0],
                    robust=True,
                    norm=colors.LogNorm(vmin=rain_lim['min'],vmax=rain_lim['max']),
                    extend='both',
                    transform=ccrs.PlateCarree())

    ds.pres_msl.plot.contour(ax=ax[2,0],
                                colors='black',
                                levels=levels,
                                alpha=0.2,
                                linewidths=2.0, 
                                transform=ccrs.PlateCarree())
    ds.pres_msl.plot.contour(ax=ax[2,0],
                               colors='black',
                               levels=levels,
                               alpha=0.9,
                               linewidths=0.25, 
                               transform=ccrs.PlateCarree())


    ####### MIXED LAYER HEIGHT ##############


    ds.mld.plot(cmap='gist_stern_r',
                ax=ax[2,1],
                robust=True,
                vmin=mld_lim['min'], vmax=mld_lim['max'],
                extend='both',
                transform=ccrs.PlateCarree())

    ax[0,1].set_ylabel('')
    ax[0,1].set_xlabel('')
    ax[1,1].set_ylabel('')
    ax[0,0].set_xlabel('')

    for i in range(0,3):
        for j in range(0,2):
            ax[i,j].set_title('')
            ax[i,j].set_extent((ds.longitude.min(), ds.longitude.max(),
                                ds.latitude.min(), ds.latitude.max()))
            # add Borders and so
            ax[i,j].coastlines(linewidth=.8, edgecolor='gray')
            ax[i,j].add_feature(BORDERS, linewidth=.8, edgecolor='black')
            # add Dutch provinces
            ax[i,j].add_geometries(prov2018, ccrs.PlateCarree(), linewidth=1.0,
                              edgecolor='darkgray', facecolor='gray', alpha=0.2)

            #Plot great-circle path
            ax[i,j].scatter(path_params['start']['lon']%360,
                            path_params['start']['lat'],
                            c='C0', edgecolors='white', marker='o',
                            transform=ccrs.PlateCarree())
            ax[i,j].scatter(path_params['end']['lon']%360,
                            path_params['end']['lat'],
                            c='C0', edgecolors='white', marker='o',
                            transform=ccrs.PlateCarree())
            ax[i,j].plot(lons, lats, linestyle='dashed', transform=ccrs.PlateCarree(),
                         path_effects=[pe.Stroke(linewidth=5, foreground='w', alpha=0.75), pe.Normal()])
            # Plot (random) sampling point
            ax[i,j].scatter(df.Longitude, df.Latitude,
                            c='white', edgecolor='black', transform=ccrs.PlateCarree())

    fig.suptitle(title_str)

    fid = '{}_map_meteo.png'.format(fid_prefix)
    fig.savefig(fid, facecolor='white')
    #fig.savefig('../'+fid, facecolor='white')

########## END PLOTTING FUNCTIONS #####################


# plotting parameters #################
temp_lim = {'min': -5.0, 'max': 5.0}
temp_lim = {'min': 10.0, 'max': 25.0}
wind_lim = {'min': 5.0, 'max': 20.0}
z_lim = {'min': 0.0, 'max': 15.0}
P_lim = {'min': 1002, 'max': 1015.0 }
mld_lim = {'min': 0.0, 'max': 5.0e3 }
tke_lim = {'min': 0.1, 'max': 5.0}
rain_lim = {'min': 1.0e-3, 'max': 100.0 }
ptke_lim = {'min': 0.0, 'max': 5.0}
#######################################

# Microbarometer sites #######################################
df = pd.DataFrame(
    {'Site': ['NL.CIA', 'NL.DBNI', 'NL.DIA', 'NL.IS311', 'NL.EXL'],
     'Location': ['Cabauw', 'De Bilt', 'Deelen', 'Dwingeloo', 'Exloo'],
     'Latitude': [51.968840, 52.098870, 52.060110, 52.811785, 52.906795],
     'Longitude': [4.927930, 5.175890, 5.887300, 6.394668, 6.865549]
    })
###############################################################

#
fid_provinces = '/Users/assink/infrasound/network/maps/NL/GMT/Provinciegrenzen_2018-shp/Provinciegrenzen_2018.shp'
prov2018 = list(shpreader.Reader(fid_provinces).geometries())
#

# parameters for great circle path
path_params = dict()
path_params['type'] = 'coordinates'
path_params['start'] = dict(lat=50.0, lon=4.0)
path_params['end'] = dict(lat=54.0, lon=7.0)
dr = 1.0e3
(lons, lats, azi, dist) = get_great_circle_path(path_params, dr)

my_atmos = HARMONIE(cycle='40')

ds_samples = []

gribs = glob.glob('cy40/*/*_GB')
for fid_grib in sorted(gribs):
    ds = read_HARMONIE(my_atmos, fid_grib)

    plot_pressure(my_atmos, ds)
    plot_meteo(my_atmos, ds)

    ds_sample = sample_model(ds, df)
    ds_samples.append(ds_sample)

ds_samples = xr.concat(ds_samples, dim='valid_time')
ds_samples.to_netcdf('dataset.nc')








