# -*- coding: utf-8 -*-
"""
Main atmosphere class

.. module:: core

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2020, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from griblib.utils.geo import get_great_circle_path

import numpy as np
import xarray as xr

# Constant of specific heat constants
gamma = 1.4
# Gas constant for an ideal gas (J/K/mol)
R = 8.31451
# Molar mass dry air and water vapor
m_air = 28.9644E-3
m_wet = 18.0153E-3
# Gas constants for dry air / water vapor J/(kg*K)
R_dry = R / m_air
R_wet = R / m_wet

# Gravitational acceleration constant (m/s^2)
g = 9.80655
cp = 1005  # J/(kg*K)

# Earth radius (meters)
earth_radius = 6367470.0
epsilon = 1e-10

class Atmosphere(object):
    def __init__(self):
        return

    def add_altitude_pressure_density(self, ds):
        # compute pressure, geopotential and geometric altitudes on model levels
        (z, pres) = self.compute_altitude_pressure(t=ds.t,
                                                   q=ds.q,
                                                   ps=ds.pres,
                                                   z0=ds.z)#.sel(hybrid=n_levels))
        dens = self.compute_density(pres, ds.t)
        ds = ds.drop_vars(['z', 'pres'])
        ds = xr.merge([z.to_dataset(), pres.to_dataset(), dens.to_dataset(), ds])
        return ds

    def compute_adiabatic_sound_speed(self, t):
        """
        Function to compute sound speed

        Parameters
        ----------
        t : `xarray.Dataset`
            Consisting absolute temperature (K)

        Returns
        -------
        cT : `xarray.Dataset`
            Contains adiabatic sound speed
        """
        cT = np.sqrt(gamma*R_dry*t)
        cT = cT.rename('cT')
        cT.attrs= {'long_name': 'Adiabatic sound speed',
                   'standard_name': 'c_T',
                   'units' : 'm/s'}
        return cT

    def compute_altitude_pressure(self, t, q, ps, z0):
        """
        Computes pressure and altitude on full-pressure levels:

        Parameters
        ----------
        t : `xarray.Dataset`
            Consisting of absolute temperature (K) on M model levels

        q : `xarray.Dataset`
            Consisting of specific humidity (kg/kg) on M model levels

        ps : `xarray.Dataset`
            Consisting of surface pressure (Pa)

        z0 : `xarray.Dataset`
            Consisting of the surface geopotential (m**2/s**2)

        Returns
        -------
        z : `xarray.Dataset`
            Contains geometric altitude (in m) on M model levels

        p : `xarray.Dataset`
            Contains full background pressure (in Pa) on M model levels
        """
        pv = t.GRIB_pv
        nlev = int(len(pv)/2 - 1)
        ps = np.broadcast_to(np.expand_dims(ps, axis=0),
                             shape=(nlev+1, *ps.shape))

        (ph, pf) = self.compute_pressure_levels(pv, ps)
        (_, z_geo) = self.compute_geo_altitudes(ph, t, q, z0)

        z = t.copy(data=z_geo).rename('z')
        z.attrs = {'long_name': 'Geometric altitude',
                   'standard_name': 'Altitude',
                   'units': 'm'
                   }

        p = t.copy(data=pf).rename('pres')
        p.attrs = {'long_name': 'Pressure',
                   'standard_name': 'Pressure',
                   'units': 'Pa'
                   }
        return (z, p)

    def compute_brunt_vaisala_freq(self, ds):
        if 'hybrid' not in ds.pres.dims:
            ds = self.add_altitude_pressure_density(ds)
        ds = self.compute_potential_temperature(ds)

        level_dim = 'hybrid'
        # Calculate differences in theta and z
        # between consecutive model levels
        dtheta = ds['theta'].diff(dim=level_dim)
        dz = ds['z'].diff(dim=level_dim)
        dz = dz.where(np.abs(dz) > epsilon, epsilon)

        # Calculate the gradient dtheta/dz
        dtheta_dz = dtheta / dz

        # Align the result to match the original dimensions
        # (shifted down by one level)
        dtheta_dz = dtheta_dz.pad({level_dim: (0, 1)},
                                   constant_values=np.nan)

        # Calculate Brunt-Vaisala Frequency squared (N^2)
        ds['N2'] = (g / ds['theta']) * dtheta_dz
        
        # The square root of N2 gives the Brunt-Vaisala Frequency (N)
        ds['N'] = np.sqrt(ds['N2'].where(ds['N2'] >= 0))
        ds['N'].attrs={'long_name': 'Brunt-Vaisala frequency',
                       'units' : 'rad/s',
                       'standard_name': 'bvf'}

        return ds

    def compute_richardson_number(self, ds):
        if 'N2' not in ds:
            ds = self.compute_brunt_vaisala_freq(ds)

        level_dim = 'hybrid'

        # Compute vertical wind shear
        du_dz = ds['u'].diff(dim=level_dim) / ds['z'].diff(dim=level_dim)
        dv_dz = ds['v'].diff(dim=level_dim) / ds['z'].diff(dim=level_dim)

        # Align the result to match original dimensions
        du_dz = du_dz.pad({level_dim: (0, 1)}, constant_values=np.nan)
        dv_dz = dv_dz.pad({level_dim: (0, 1)}, constant_values=np.nan)

        wind_shear_squared = du_dz**2 + dv_dz**2

        # Richardson number
        Ri = ds['N2'] / (wind_shear_squared + epsilon)  # prevent division by zero
        Ri = Ri.where(wind_shear_squared > 0)  # mask where denominator is essentially zero

        ds['Ri'] = Ri
        ds['Ri'].attrs = {
            'long_name': 'Gradient Richardson number',
            'units': 'dimensionless',
            'standard_name': 'richardson_number'
        }

        return ds


    def compute_density(self, pres, t):
        """
        Compute atmospheric density assuming atmosphere behaves like an ideal gas

        Parameters
        ----------
        pf : `numpy.array`
            Array consisting of M full pressure level values (Pa)

        T : `numpy.array`
            Array consisting of absolute temperature (K) at M model levels

        Returns
        -------
        dens : `numpy.array`
            Atmospheric density on M model levels (kg m**-3)
        """
        #dens = self._compute_density(pres, t)
        dens = pres / (R_dry * t)
        dsd = t.copy(data=dens).rename('den')
        dsd.attrs = {'long_name': 'Density',
                    'standard_name': 'Density',
                    'units': 'kg m**-3'
                    }
        return dsd

    def compute_geo_altitudes(self, ph, T, q, z_pot_gnd):
        """
        Computes geopotential and geometric altitudes given half-pressure levels,
        and full level values of pressure, temperature and specific humidity:

        Parameters
        ----------
        ph : `numpy.array`
            Array consisting of M+1 half pressure level values (in Pa)

        T : `numpy.array`
            Array consisting of absolute temperature (in K) at M model levels

        q : `numpy.array`
            Array consisting of specific humidity (in kg/kg) at M model levels

        z_pot_gnd : `numpy.array`
            Array consisting of geopotential value (in m**2/s**2) on the ground

        Returns
        -------
        z_pot : `numpy.array`
            Geopotential altitude on M model levels (in m)

        z_geo : `numpy.array`
            Geometric altitude on M model levels (in m)
        """
        # Initialize arrays
        nlev = T.shape[0]

        z_pot = np.zeros(T.shape)
        dlogP = np.zeros(T.shape)
        alpha = np.zeros(T.shape)

        # level 1 (top of atmosphere)
        dlogP[0] = np.log(ph[1] / 0.1)
        alpha[0] = np.log(2.)
        # levels 2 - M
        ph_lev = ph[1:-1]
        ph_levplusone = ph[2:]
        dlogP[1:] = np.log(ph_levplusone / ph_lev)
        alpha[1:] = 1. - ((ph_lev / (ph_levplusone - ph_lev)) * dlogP[1:])

        Tv = T*(1+(R_wet/R_dry-1)*q)
        RTv = R_dry*Tv

        # Compute geopotential at all model levels, starting from ground level
        z_pot_h = np.copy(z_pot_gnd)
        for i in range(0,nlev):
            k = nlev - 1 - i
            z_pot[k] = z_pot_h + alpha[k] * RTv[k]
            z_pot_h += RTv[k] * dlogP[k]
            
        (z_pot, z_geo) = self.geopotential_to_altitudes(z_pot)
        return (z_pot, z_geo)

    def compute_mean_sea_level_pressure(self, pres_surface, z_surface, t_surface):
        """
        Function to compute mean sea level pressure using simple barometric formula
        for standard atmosphere. This approximation does not hold for flat terrain.

        Parameters
        ----------
        pres_surface : `xarray.Dataset`
            XArray Dataset containing surface pressure data (Pa)

        z_surface : `xarray.Dataset`
            XArray Dataset containing surface elevation (m)

        t_surface : `xarray.Dataset`
            XArray Dataset containing surface temperature (K)

        Returns
        -------
        msl : `xarray.Dataset`
            XArray Dataset containing mean sea level corrected pressure data (Pa)
        """
        L = -0.0065  # Lapse rate [K/m]
        T_MSL = t_surface - L*z_surface
        base = T_MSL / (T_MSL + L*z_surface)
        exp = g*m_air / (R*L)

        msl = (pres_surface / np.power(base,exp)).rename('msl')
        msl.attrs= {'long_name': 'mean sea level pressure',
                    'standard_name': 'mean_sea_level_pressure',
                    'units' : 'Pa'}
        return msl

    def compute_potential_temperature(self, ds):
        p0 = ds.pres.sel(hybrid=len(ds.hybrid))
        ds['theta'] = ds['t'] * ( p0 / ds['pres'])**(R_dry/cp)
        ds['theta'].attrs={'long_name': 'Potential temperature',
                           'units' : 'K',
                           'standard_name': 'theta'}
        return ds

    def compute_pressure_levels(self, pv, ps):
        """
        Computes half- and full pressure levels given a surface pressure
        and GRIB PV coefficients

        Parameters
        ----------
        pv : `numpy.array`
            2*(NLEV + 1) PV coefficients from GRIB file

        ps : `float` or `numpy.array`
            Surface pressure value (in Pa)

        Returns
        -------
        ph : `numpy.array`
            Array consisting of M + 1 half pressure level values (in Pa)

        pf : `numpy.array`
            Array consisting of M full pressure level values (in Pa)
        """
        nlev = int(len(pv)/2 - 1)
        pv = np.array(pv).reshape(2, nlev+1)
        a_coeff=pv[0,:]
        b_coeff=pv[1,:]

        # Compute half-pressure levels.
        newshape = np.ones(ps.ndim, dtype=np.int32)
        newshape[0] = len(a_coeff)
        a_coeff = np.broadcast_to(a_coeff.reshape(newshape), shape=ps.shape)
        b_coeff = np.broadcast_to(b_coeff.reshape(newshape), shape=ps.shape)
        ph = a_coeff + b_coeff*ps

        # Compute full-pressure levels
        ph_1 = ph[0:-1]
        ph_2 = ph[1:]
        pf = 0.5*(ph_1 + ph_2)
        return (ph, pf)

    def compute_turbulent_pressure(self, ds):
        ds['pres_turb'] = (ds.tke * ds.den).rename('pres_turb')
        ds.pres_turb.attrs={'long_name': 'Turbulent pressure',
                            'units' : 'Pa',
                            'standard_name': 'Turbulent pressure'}
        return ds

    def compute_wind_speed_and_direction(self, ds):
        ds['wind_speed'] = np.sqrt(ds['u']**2 + ds['v']**2)
        ds['wind_speed'].attrs['units'] = 'm/s'
        ds['wind_speed'].attrs['description'] = 'Horizontal wind speed'

        wind_dir_rad = np.arctan2(ds['v'], ds['u'])
        ds['wind_direction'] = (270 - np.degrees(wind_dir_rad)) % 360
        ds['wind_direction'].attrs['units'] = 'degrees'
        ds['wind_direction'].attrs['description'] = 'Horizontal wind direction'
        return ds

    def extract_gcp_slice(self, ds, path_params, dr=100.0e3):
        """
        Function to extract vertical slice out of model along great circle path

        Parameters
        ----------
        ds : `xarray.Dataset`
            XArray Dataset containing original model data

        path_params : `dict`
            Dictionary containing parameters on how to form great circle path

        dr : `float`
            Spacing between great circle points (in meter)

        Returns
        -------
        ds_gcp : `xarray.Dataset`
            XArray Dataset with interpolated model data on gcp path
        """
        # Determine great circle path parameters
        (lons, lats, azi, dist) = get_great_circle_path(path_params, dr)

        xlon = xr.DataArray(lons, dims='distance')
        ylat = xr.DataArray(lats, dims='distance')

        # Interpolate XArray DataSet along great circle path
        ds.load()  # fix otherwise it won't work on recent XArray versions
        ds_gcp = ds.interp(longitude=xlon, latitude=ylat)
        
        # Assign distance along great circle path to the distance dimension
        ds_gcp = ds_gcp.assign_coords(distance=dist)
        ds_gcp['distance'].attrs = {'long_name': 'distance', 
                                    'units': 'm', 
                                    'positive': 'right', 
                                    'stored_direction': 'increasing',
                                    'standard_name': 'distance'}
        # Store azimuth along great circle path as XArray coordinate
        ds_gcp['azimuth'] = xr.DataArray(azi, coords=[dist], dims=['distance'])
        ds_gcp['azimuth'].attrs = {'long_name': 'azimuth', 
                                   'units': 'degrees', 
                                   'positive': 'right', 
                                   'stored_direction': 'increasing',
                                   'standard_name': 'azimuth'}
        ds_gcp = ds_gcp.set_coords('azimuth')
        return ds_gcp

    def geopotential_to_altitudes(self, z_pot):
        """
        Function to convert geopotential (m**/s**2) to geopotential altitude (m)
        and geometric altitude (m)

        Parameters
        ----------
        z_pot : `xarray.Dataset`
            XArray Dataset containing geopotential (m**2/s**2)

        Returns
        -------
        z_pot : `xarray.Dataset`
            XArray Dataset containing geopotential altitude (m)

        z_geo : `xarray.Dataset`
            XArray Dataset containing geometric altitude (m)
        """
        # Convert geopotential (m**2/s**2) to geopotential altitude (m)
        z_pot /= g
        # Convert geopotential altitude (m) to geometric altitude (m)
        z_geo = earth_radius*z_pot / (earth_radius - z_pot)
        return (z_pot, z_geo)

    def omega_to_w(self, density, omega):
        """
        Scale vertical velocity from Pa/s to m/s, which is used
        in certain models, i.e. ECMWF, AROME (AT) and HIRLAM (NL)
        """
        w = -omega / (g*density)
        w = omega.copy(data=w).rename('w')
        w.attrs = {'long_name': 'Vertical velocity',
                    'standard_name': 'Vertical velocity',
                    'units': 'm s**-1'
                    }
        return w

    def _read_grib(self, fid_grib, list_keys, filter_keys, verbose):
        """
        Function to read HARMONIE GRIB file and return XArray dataset.

        Parameters
        ----------

        fid_grib : `str`
            Filename of GRIB file

        list_keys : list of `str`
            Type of GRIB parameters to be read, such as PV coefficients

        filter_keys : `dict`
            Dictionary with information necessry to read a variable. Keys can include
            `typeOfLevel`, `shortName` and `stepType`.

        Returns
        -------
        ds : `xarray.Dataset`
            XArray Dataset containing original model data
        """

        backend_args = {'read_keys': list_keys, 'filter_by_keys': filter_keys}
        # if verbose is False:
        #     backend_args['errors'] = 'ignore'

        ds = xr.open_dataset(fid_grib, engine='cfgrib', backend_kwargs=backend_args,
                             decode_timedelta=True)

        return ds

    def plot_prefix(self, ds):
        t_fcst=(ds.valid_time).dt.strftime('%Y%m%d-%H').values
        title_str = f'{self.model}{self.cycle} forecast {t_fcst} UTC'
        fid_prefix = f'{self.model}{self.cycle}_{t_fcst}'
        return (t_fcst, title_str, fid_prefix)

    def read_grib(self, fid_grib, request, pv_coefficients=True, verbose=True, **kwargs):
        """
        Wrapper function to combine multiple level types in one dataset

        Parameters
        ----------

        fid_grib : `str`
            Filename of GRIB file

        request : list of dictionaries
            Dictionary entries should consist of 'level_type' 'var_list'
            entries. 'var_list' is a list of model parameters to be read
            and its entries should correspond to the 'shortName' parameter.
            By default all known variables are attempted to be read.
            A parameter is skipped if is not present in the file.

        pv_coefficients : `Boolean`
            Add GRIB PV coefficients to resulting XArray DataSet

        Returns
        -------
        ds : `xarray.Dataset`
            XArray Dataset containing original model data
        """
        print('*'*80)
        print(f'Reading {self.model} cy{self.cycle} GRIB file [ {fid_grib} ]')
        print('')

        ds = []
        list_keys = []

        if pv_coefficients:
            print('Reading GRIB PV coefficients')
            list_keys.append('pv')

        for item in request:
            if 'var_list' in item:
                for shortName in item['var_list']:
                    filter_keys = {'typeOfLevel': item['level_type'],
                                   'shortName': shortName}

                    msg = (f'Reading {filter_keys["shortName"]} '
                           f'on {filter_keys["typeOfLevel"]} level')

                    if 'step_type' in item:
                        filter_keys['stepType'] = item['step_type']
                        msg += ' (stepType {})'.format(item['step_type'])
                    if 'level' in item:
                        filter_keys['level'] = item['level']
                        msg += ' (level {})'.format(item['level'])

                    print(msg)
                    try:
                        dss = self._read_grib(fid_grib, list_keys,
                                              filter_keys, verbose)
                        ds.append(dss)
                    except ValueError as e:
                        print(e)
                        pass
                    except KeyError:
                        msg = (' - **ERROR**: variable {variable} not found.'
                               ' Skipping...'.format(variable=shortName))
                        print (msg)
                        pass

            # 'Greedy style' request without variable list
            else:
                filter_keys = {'typeOfLevel': item['level_type']}
                msg = ('Reading all variables on {level_type} level'.format(
                       level_type=filter_keys['typeOfLevel']))
                if 'step_type' in item:
                    filter_keys['stepType'] = item['step_type']
                    msg += ' (stepType {})'.format(item['step_type'])
                if 'level' in item:
                    filter_keys['level'] = item['level']
                    msg += ' (level {})'.format(item['level'])

                print(msg)
                try:
                    dss = self._read_grib(fid_grib, list_keys,
                                          filter_keys, verbose)
                    ds.append(dss)
                except Exception as e:
                    print(e)
                    msg = (' - ERROR reading in level type {level_type}.'
                        .format(level_type=filter_keys['typeOfLevel']))

        ds = xr.merge(ds, **kwargs)
        return ds

    def rotate_to_along_cross_winds(self, u, v, azimuth):
        """
        Compute along-track and cross-track winds on all model levels
     
        Parameters
        ----------
        u : `xarray.Dataset`
            XArray Dataset containing zonal wind data (m/s)

        v : `xarray.Dataset`
            XArray Dataset containing meridional wind data (m/s)

        azimuth : `xarray.Dataset`
            XArray Dataset containing propagation azimuth values (deg)

        Returns
        -------
        wa : `xarray.Dataset`
            XArray Dataset including along-track wind

        wc : `xarray.Dataset`
            XArray Dataset including cross-track wind
        """
        wa = (u * np.sin(np.deg2rad(azimuth)) + \
              v * np.cos(np.deg2rad(azimuth)))
        wa = wa.rename('wa')
        wa.attrs = {'long_name': 'Along track wind speed',
                    'standard_name': 'w_a',
                    'units' : 'm/s'}

        wc = (v * np.sin(np.deg2rad(azimuth)) - \
              u * np.cos(np.deg2rad(azimuth)))
        wc = wc.rename('wc')
        wc.attrs = {'long_name': 'Cross track wind speed',
                    'standard_name': 'w_c',
                    'units' : 'm/s'}
        return (wa, wc)

    def wrap_around_360deg_dataset(self, ds, lon_0=0, lon_1=360):
        """
        Function add another entry at 360 degrees.
        This facilitates plotting without a gap.

        Parameters
        ----------
        ds : `xarray.Dataset`
            XArray Dataset containing original model data
        """
        # Add a 360 longitude entry if the data wraps
        #if ds.longitude.min() == 0.0 and ds.longitude.max() != 360.0:
        lon0 = ds.sel(longitude=lon_0)
        lon0['longitude'] = lon_1
        ds = xr.concat([ds, lon0], 'longitude')
        return ds
