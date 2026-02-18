# -*- coding: utf-8 -*-
"""
Helper routines to sample model data

.. module:: sampling

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2020, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
import xarray as xr
import pandas as pd

def inventory_to_dataframe(inv):
    records = []

    for network in inv:
        for station in network.stations:
            net_sta = f"{network.code}.{station.code}"
            location = station.site.name or ""
            latitude = station.latitude
            longitude = station.longitude

            records.append({
                "Site": net_sta,
                "Location": location,
                "Latitude": latitude,
                "Longitude": longitude
            })

    return pd.DataFrame(records)

def sample_model(ds, df):
    ds_samples = []

    pts_lat = xr.DataArray(df.Latitude, dims='site_index')
    pts_lon = xr.DataArray(df.Longitude, dims='site_index')

    ds.load()
    ds_samples = ds.interp(longitude=pts_lon, latitude=pts_lat)

    ds_samples = ds_samples.assign_coords(site_code=("site_index", df.Site.values))
    ds_samples = ds_samples.assign_coords(site_location=("site_index", df.Location.values))
    return ds_samples