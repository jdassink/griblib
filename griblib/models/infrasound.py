# -*- coding: utf-8 -*-
"""
Infrasound helper functions

.. module:: infrasound

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2025, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from griblib.utils.geo import get_great_circle_path
from griblib import Atmosphere

import numpy as np
import xarray as xr
import pyproj


# Constant of specific heat constants
gamma = 1.4
# Gas constant for an ideal gas (J/K/mol)
R = 8.31451
# Molar mass dry air and water vapor
m_air = 28.9644E-3
m_wet = 18.0153E-3
# Gas constants for dry air / water vapor
R_dry = R / m_air
R_wet = R / m_wet

# Gravitational acceleration constant (m/s^2)
g = 9.80655
# Earth radius (meters)
earth_radius = 6367470.0

class Infrasound(Atmosphere):
    def __init__(self, **kwargs):
        Atmosphere.__init__(self)

        return
    
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

    def compute_azimuth_grid(self, ds, src_lat, src_lon):
        """
        Compute azimuth grid
        """
        g = pyproj.Geod(ellps='WGS84')

        X, Y = np.meshgrid(ds.longitude, ds.latitude)
        SX = np.ones(X.shape) * src_lon
        SY = np.ones(Y.shape) * src_lat
        #(_, azis, _) = g.inv(X, Y, SX, SY, radians=False)
        (_, bazis, _) = g.inv(SX, SY, X, Y, radians=False)

        azis = 180 + bazis
        #azis = 0.0 + azis % 360
        # coords=dict(
        #         lon=ds.coords['longitude'],
        #         lat=ds.coords['latitude'],
        #         time=ds.coords['time'],
        #         step=ds.coords['step'],
        #         valid_time=ds.coords['valid_time']
        #         )
        azimuth = xr.DataArray(data=azis, name='azimuth',
                               dims=('latitude', 'longitude'))
        azimuth.attrs = {'long_name': 'Propagation azimuth',
                         'standard_name': 'azimuth',
                         'units': 'deg'}
        return azimuth

    def effective_sound_speed_ratio(self, ce, c0):
        """
        Compute adiabatic sound speed, along-track and cross-track wind,
        and the effective sound speed (ratio) on model levels.
        
        Parameters
        ----------
        ce : `xarray.Dataset`
            XArray Dataset containing effective sound speed (m/s)

        c0 : `xarray.Dataset`
            XArray Dataset containing effective sound speed conditions
            at the receiver level (m/s)

        Returns
        -------
        cr : `xarray.Dataset`
            XArray Dataset containing effective sound speed ratio (-)
        """
        cr = (ce / c0).rename('cr')
        cr.attrs = {'long_name': 'Effective sound speed ratio',
                    'standard_name': 'c_eff ratio',
                    'units' : '-'}
        return cr

    def infrasound_propagation_conditions(self, t, u, v, azimuth):
        """
        Compute adiabatic sound speed, along-track and cross-track wind,
        and the effective sound speed (ratio) on model levels.
        
        Parameters
        ----------
        t : `xarray.Dataset`
            XArray Dataset containing temperature (K)

        u : `xarray.Dataset`
            XArray Dataset containing zonal wind (m/s)

        v : `xarray.Dataset`
            XArray Dataset containing meridional wind (m/s)

        azimuth : `xarray.Dataset`
            XArray Dataset containing propagation azimuth values (deg)

        Returns
        -------
        is : `xarray.Dataset`
            XArray Dataset including infrasound propagation parameters
        """
        cT = self.compute_adiabatic_sound_speed(t)
        (wa, wc) = self.rotate_to_along_cross_winds(u, v, azimuth)

        ce = (cT + wa).rename('ce')
        ce.attrs = {'long_name': 'Effective sound speed',
                   'standard_name': 'c_eff',
                   'units' : 'm/s'}
        return (cT, wa, wc, ce)