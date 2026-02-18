# -*- coding: utf-8 -*-
"""
ECMWF atmosphere model class

.. module:: ecmwf

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
import numpy as np
import pandas as pd
import xarray as xr

class ECMWF(Atmosphere):
    def __init__(self, cycle=None, **kwargs):
        Atmosphere.__init__(self)
        self.model = 'ECMWF'
        try:
            self.cycle = int(cycle)
        except Exception as e:
            print(e)
            self.cycle = ''
            pass
        return

    def attach_geopotential_field(self, ds, fid_grib_z, **kwargs):
        """
        Function to attach geopotential to XArray Dataset without geopotential.
        This typically occurs when reading ECMWF forecast GRIB files.

        It is required that the grid of the geopotential is sampled on the 
        same grid as is present in the original Dataset.

        Parameters
        ----------
        ds : `xarray.Dataset`
            XArray Dataset containing original model data

        fid_grib_z : `str`
            Filename of GRIB file with geopotential

        Returns
        -------
        ds : `xarray.Dataset`
            XArray Dataset containing original model data
        """
        request = [
            {'level_type': 'hybrid', 'var_list': ['z']}
        ]

        dsz = self.read_grib(fid_grib_z, request)
        dsz['step'] = ds['step']
        dsz['valid_time'] = ds['valid_time']

        print('Attaching geopotential field')
        ds = xr.merge([dsz, ds])
        return ds

    def convert_lnsp_to_sp(self, lnsp):
        """
        Function to compute surface pressure from ECMWF lnsp field

        Parameters
        ----------
        lnsp : `xarray.Dataset`
            Contains logarithm of surface pressure (ln[Pa])

        Returns
        -------
        sp : `xarray.Dataset`
            Contains surface pressure (Pa)
        """
        sp = np.exp(lnsp).rename('sp')
        sp.attrs= {'long_name': 'surface pressure',
                   'standard_name': 'surface_pressure',
                   'units' : 'Pa'}
        return sp

    def ecmwf_L137_coefficients(self):
        """
        Returns A and B coefficients and temperature as tabulated on ECMWF website:
        https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels

        Returns
        -------
        pv : `numpy.array`
            Two dimensional array consisting of A and B coefficents

        ps : `numpy.array`
            Surface pressure value (in Pa)

        T : `numpy.array`
            Array consisting of temperature values in ECMWF standard atmosphere

        q : `numpy.array`
            Array consisting of specific humidity (in kg/kg) at M model levels
        """
        # hard coded coeffients:
        ah_coeff = np.array([0.000000, 2.000365, 3.102241, 4.666084, 6.827977,
                            9.746966, 13.605424, 18.608931, 24.985718, 32.985710,
                            42.879242, 54.955463, 69.520576, 86.895882,
                            107.415741, 131.425507, 159.279404, 191.338562,
                            227.968948, 269.539581, 316.420746, 368.982361,
                            427.592499, 492.616028, 564.413452, 643.339905,
                            729.744141, 823.967834, 926.344910, 1037.201172,
                            1156.853638, 1285.610352, 1423.770142, 1571.622925,
                            1729.448975, 1897.519287, 2076.095947, 2265.431641,
                            2465.770508, 2677.348145, 2900.391357, 3135.119385,
                            3381.743652, 3640.468262, 3911.490479, 4194.930664,
                            4490.817383, 4799.149414, 5119.895020, 5452.990723,
                            5798.344727, 6156.074219, 6526.946777, 6911.870605,
                            7311.869141, 7727.412109, 8159.354004, 8608.525391,
                            9076.400391, 9562.682617, 10065.978516, 10584.631836,
                            11116.662109, 11660.067383, 12211.547852, 12766.873047,
                            13324.668945, 13881.331055, 14432.139648, 14975.615234,
                            15508.256836, 16026.115234, 16527.322266, 17008.789063,
                            17467.613281, 17901.621094, 18308.433594, 18685.718750,
                            19031.289063, 19343.511719, 19620.042969, 19859.390625,
                            20059.931641, 20219.664063, 20337.863281, 20412.308594,
                            20442.078125, 20425.718750, 20361.816406, 20249.511719,
                            20087.085938, 19874.025391, 19608.572266, 19290.226563,
                            18917.460938, 18489.707031, 18006.925781, 17471.839844,
                            16888.687500, 16262.046875, 15596.695313, 14898.453125,
                            14173.324219, 13427.769531, 12668.257813, 11901.339844,
                            11133.304688, 10370.175781, 9617.515625, 8880.453125,
                            8163.375000, 7470.343750, 6804.421875, 6168.531250,
                            5564.382813, 4993.796875, 4457.375000, 3955.960938,
                            3489.234375, 3057.265625, 2659.140625, 2294.242188,
                            1961.500000, 1659.476563, 1387.546875, 1143.250000,
                            926.507813, 734.992188, 568.062500, 424.414063,
                            302.476563, 202.484375, 122.101563, 62.781250,
                            22.835938, 3.757813, 0.000000, 0.000000])
        bh_coeff = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0.000007, 0.000024, 0.000059, 0.000112, 0.000199,
                            0.000340, 0.000562, 0.000890, 0.001353, 0.001992,
                            0.002857, 0.003971, 0.005378, 0.007133, 0.009261,
                            0.011806, 0.014816, 0.018318, 0.022355, 0.026964,
                            0.032176, 0.038026, 0.044548, 0.051773, 0.059728,
                            0.068448, 0.077958, 0.088286, 0.099462, 0.111505,
                            0.124448, 0.138313, 0.153125, 0.168910, 0.185689,
                            0.203491, 0.222333, 0.242244, 0.263242, 0.285354,
                            0.308598, 0.332939, 0.358254, 0.384363, 0.411125,
                            0.438391, 0.466003, 0.493800, 0.521619, 0.549301,
                            0.576692, 0.603648, 0.630036, 0.655736, 0.680643,
                            0.704669, 0.727739, 0.749797, 0.770798, 0.790717,
                            0.809536, 0.827256, 0.843881, 0.859432, 0.873929,
                            0.887408, 0.899900, 0.911448, 0.922096, 0.931881,
                            0.940860, 0.949064, 0.956550, 0.963352, 0.969513,
                            0.975078, 0.980072, 0.984542, 0.988500, 0.991984,
                            0.995003, 0.997630, 1.000000])
        #pv = np.array([ah_coeff, bh_coeff])
        pv = np.concatenate((ah_coeff, bh_coeff), axis=None)

        T = np.array([198.05, 209.21, 214.42, 221.32, 228.06, 234.56, 240.83,
                    246.87, 252.71, 258.34, 263.78, 269.04, 270.65, 270.65,
                    269.02, 264.72, 260.68, 256.89, 253.31, 249.94, 246.75,
                    243.73, 240.86, 238.14, 235.55, 233.09, 230.74, 228.60,
                    227.83, 227.09, 226.38, 225.69, 225.03, 224.39, 223.77,
                    223.17, 222.60, 222.04, 221.50, 220.97, 220.46, 219.97,
                    219.49, 219.02, 218.57, 218.12, 217.70, 217.28, 216.87,
                    216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65,
                    216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65,
                    216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65,
                    216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.74,
                    218.62, 220.51, 222.40, 224.29, 226.18, 228.08, 229.97,
                    231.86, 233.75, 235.64, 237.53, 239.42, 241.31, 243.20,
                    245.09, 246.98, 248.86, 250.75, 252.63, 254.50, 256.35,
                    258.17, 259.95, 261.68, 263.37, 265.00, 266.57, 268.08,
                    269.52, 270.90, 272.21, 273.45, 274.63, 275.73, 276.77,
                    277.75, 278.66, 279.52, 280.31, 281.05, 281.73, 282.37,
                    282.96, 283.50, 284.00, 284.47, 284.89, 285.29, 285.65,
                    285.98, 286.28, 286.56, 286.81, 287.05, 287.26, 287.46,
                    287.64, 287.80, 287.95, 288.09])
        T = xr.DataArray(data=T,
                         name='t',
                         dims=('hybrid'),
                         attrs=dict(long_name='temperature',
                                    standard_name='temperature',
                                    units='K'))

        nlev = len(T)
        # broadcast variables to right dimensions
        #ps = np.broadcast_to(np.expand_dims(ps, axis=0), shape=(nlev+1, *ps.shape))
        ps = 101325.0
        ps = np.broadcast_to(np.expand_dims(ps, axis=0), shape=(nlev+1,))
        ps = xr.DataArray(data=ps,
                          name='sp',
                          dims=('hybrid'),
                          attrs=dict(long_name='surface pressure',
                                     standard_name='surface_pressure',
                                     units='Pa'))

        q = T.copy()*0
        q.attrs = dict(long_name='specific humidity',
                       standard_name='specific_humidity',
                       units='kg/kg')
        return(pv, ps, T, q)

    def ecmwf_L137_coefficients_from_file(self, fid):
        """
        Read A and B coefficients and temperature from values on ECMWF website:
        https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels

        Parameters
        ----------
        fid : `str`
            Filename that contains coefficients, as tabulated on ECMWF website

        Returns
        -------
        pv : `numpy.array`
            Two dimensional array consisting of A and B coefficents

        ps : `numpy.array`
            Surface pressure value (in Pa)

        T : `numpy.array`
            Array consisting of temperature values in ECMWF standard atmosphere

        q : `numpy.array`
            Array consisting of specific humidity (in kg/kg) at M model levels
        """
        # cols = ['n', 'a_coeff', 'b_coeff',
        #         'ph', 'pf', 'z_pot', 'z_geo',
        #         'T', 'dens']
        cols = ['a_coeff', 'b_coeff', 'ph', 'T']
        df = pd.read_table(fid, skiprows=2, names=cols, usecols=(1, 2, 3, 7), 
                            dtype=np.float64, na_values='-')
        ah_coeff = df['a_coeff'].values
        bh_coeff = df['b_coeff'].values
        pv = np.concatenate((ah_coeff, bh_coeff), axis=None)

        T = df['T'].values[1:]
        T = xr.DataArray(data=T,
                         name='t',
                         dims=('hybrid'),
                         attrs=dict(long_name='temperature',
                                    standard_name='temperature',
                                    units='K'))

        nlev = len(T)
        # broadcast variables to right dimensions
        #ps = np.broadcast_to(np.expand_dims(ps, axis=0), shape=(nlev+1, *ps.shape))
        ps = df['ph'].values[-1]*1e2
        ps = np.broadcast_to(np.expand_dims(ps, axis=0), shape=(nlev+1,))
        ps = xr.DataArray(data=ps,
                          name='sp',
                          dims=('hybrid'),
                          attrs=dict(long_name='surface pressure',
                                     standard_name='surface_pressure',
                                     units='Pa'))

        q = T.copy()*0
        q.attrs = dict(long_name='specific humidity',
                       standard_name='specific_humidity',
                       units='kg/kg')
        return(pv, ps, T, q)