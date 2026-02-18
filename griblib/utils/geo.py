# -*- coding: utf-8 -*-
"""
Helper routines for geographical operations

.. module:: geo

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2016-2020, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
import numpy as np
import pyproj

def degrees2kilometers(degrees, radius=6371):
    """
    Original ObsPy function
    Convenience function to convert (great circle) degrees to kilometers
    assuming a perfectly spherical Earth.

    :type degrees: float
    :param degrees: Distance in (great circle) degrees
    :type radius: int, optional
    :param radius: Radius of the Earth used for the calculation.
    :rtype: float
    :return: Distance in kilometers as a floating point number.

    .. rubric:: Example

    >>> from obspy.geodetics import degrees2kilometers
    >>> degrees2kilometers(1)
    111.19492664455873
    """
    return degrees * (2.0 * radius * np.pi / 360.0)


def kilometers2degrees(kilometer, radius=6371):
    """
    Original ObsPy function
    Convenience function to convert kilometers to degrees assuming a perfectly
    spherical Earth.

    :type kilometer: float
    :param kilometer: Distance in kilometers
    :type radius: int, optional
    :param radius: Radius of the Earth used for the calculation.
    :rtype: float
    :return: Distance in degrees as a floating point number.

    .. rubric:: Example

    >>> from obspy.geodetics import kilometers2degrees
    >>> kilometers2degrees(300)
    2.6979648177561915
    """
    return kilometer / (2.0 * radius * np.pi / 360.0)


def inverse_transform(start, end):
    """
    Helper function. Returns (back)azimuth and distance on WGS-84 ellipsoid

    Parameters
    ----------
    start : `dict`
        Dictionary w/ latitude ('lat') and longitude ('lon') of start point

    end : `dict`
        Dictionary w/ latitude ('lat') and longitude ('lon') of end point

    Returns
    -------
    azi : `float`
        Azimuth (between 0-360 deg) from start to end point

    bazi : `float`
        Back azimuth (between 0-360 deg) from end to start point

    dist : `float`
        Distance (in meters) between start end end point
    """
    g = pyproj.Geod(ellps='WGS84')
    (azi, bazi, dist) = g.inv(start['lon'], start['lat'],
                                end['lon'], end['lat'], radians=False)
    return(azi%360,bazi%360, dist)


def get_great_circle_path(path_params, dr):
    """
    Helper function, returns lons, lats and aux info along great circle path

    Parameters
    ----------
    path_params : `dict`
        Dictionary containing parameters on how to form great circle path

    dr : `float`
        Spacing between great circle points (in meter)

    Returns
    -------
    lons : `numpy.array`
        Longitudes (between 0-360 deg) along great circle path

    lats : `numpy.array`
        Latitudes (between 0-360 deg) along great circle path

    azis : `numpy.array`
        Azimuth (in degrees) along the great circle path

    dist : `numpy.array`
        Distances (in meters) along the great circle path
    """
    try:
        assert(path_params['type']=='coordinates' or 
               path_params['type']=='range-azimuth')
    except AssertionError:
        msg = ("params['type'] must be 'coordinates' or 'range-azimuth'")
        print(msg)
        raise

    g = pyproj.Geod(ellps='WGS84')

    start = path_params['start']
    if path_params['type'] == 'coordinates':
        end = path_params['end']
        (_, _, dist) = inverse_transform(start, end)

    elif path_params['type'] == 'range-azimuth':
        dist = path_params['range']
        azi = path_params['azimuth']
        (elo, ela, _) = g.fwd(start['lon'], start['lat'], azi, dist)
        end = dict(lat=ela, lon=elo)

    n_gcp = int(dist/dr)
    # Compute latitude-longitude points along great circle path
    lonlats = g.npts(start['lon'], start['lat'], end['lon'], end['lat'],
                        n_gcp, radians=False)
    lonlats.insert(0, (start['lon'], start['lat']))
    lonlats.append((end['lon'], end['lat']))
    lonlats = np.array(lonlats)
    lons = lonlats[:,0]%360
    lats = lonlats[:,1]

    # Compute azimuth and distance vector along great circle path
    (azis, _, dist) = g.inv(lons[0:-1], lats[0:-1], lons[1:], lats[1:],
                            radians=False)
    azis = np.insert(azis, 0, azis[0]-(azis[1]-azis[0]))
    dist = np.insert(dist, 0, 0.0)
    dist = np.cumsum(dist)
    return (lons, lats, azis, dist)