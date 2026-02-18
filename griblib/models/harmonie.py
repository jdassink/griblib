# -*- coding: utf-8 -*-
"""
HARMONIE atmosphere model class

.. module:: harmonie

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2020, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from griblib.models.core import Atmosphere

class HARMONIE(Atmosphere):
    def __init__(self, cycle=None, **kwargs):
        Atmosphere.__init__(self)
        self.model = 'HARMONIE'
        try:
            self.cycle = int(cycle)
        except Exception as e:
            print(e)
            self.cycle = ''
            pass

        return

    def read_tke(self, fid_grib):
        """
        Reads and processes GRIB data to extract turbulence kinetic energy (TKE) 
        and related atmospheric variables.

        Parameters
        ----------
        fid_grib : str or file-like
            Path to the GRIB file or file object containing the meteorological data.

        Returns
        -------
        xarray.Dataset
            A dataset containing variables from multiple atmospheric levels, 
            including:
            - Hybrid levels: geopotential height ('z'), temperature ('t'), 
            eastward and northward wind components ('u', 'v'), vertical wind ('tw'), 
            specific humidity ('q'), and turbulence kinetic energy ('tke').
            - Height above ground level: surface pressure ('pres') and 
            mixed layer depth ('mld').

        Notes
        -----
        The method also computes and adds altitude, pressure, and air density 
        fields to the dataset using `add_altitude_pressure_density`.

        Commented-out code hints at the possibility of including sea-level pressure 
        ('pres_msl') in future versions.
        """
        request = [
            {'level_type': 'hybrid', 'var_list': ['z', 't', 'u', 'v', 'tw', 'q', 'tke']},
            {'level_type': 'heightAboveGround', 'var_list': ['pres'], 'level': 0},
            {'level_type': 'heightAboveGround', 'var_list': ['mld']}
        ]
        ds = self.read_grib(fid_grib, request, compat='override')
        ds = self.add_altitude_pressure_density(ds)

        request = [
            {'level_type': 'heightAboveSea', 'var_list': ['pres'], 'level': 0},
        ]
        ds_msl = self.read_grib(fid_grib, request, compat='override')
        ds['pres_msl'] = ds_msl['pres'] / 1e2
        return ds