# -*- coding: utf-8 -*-
"""
Main atmosphere class

.. module:: mod_ascii

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2020, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
import numpy as np

def write_profile(ds, fid, model='ecmwf', header=True):
    """
    This routine writes the profile to a standard 1-D atmosphere
    format that can be used with NCPAprop.

    Parameters
    ----------
    ds : `xarray.Dataset`
        XArray Dataset containing 1-D atmosphere model

    fid : `str`
        Name of file to write to
    """
    print("*"*80)
    print("Writing 1-D profile to [ {} ]".format(fid))

    if header:
        if model == 'ecmwf' or 'harmonie':
            z0 = ds.z.min()
            z = (ds.z - z0).data
        elif model == 'g2s':
            z0 = ds.z0
            z = ds.z.data

        header = (f'% 0, Z0, m, {z0.data:.1f}\n'
                f'% 1, Z, km\n'
                f'% 2, U, m/s\n'
                f'% 3, V, m/s\n'
                f'% 4, W, m/s\n'
                f'% 5, T, degK\n'
                f'% 6, RHO, g/cm3\n'
                f'% 7, P, mbar')
    else:
        header = ''
        z = ds.z.data

    u = ds.u.data
    v = ds.v.data

    if model == 'ecmwf':
        try:
            w = ds.w.data
        except:
            w = np.zeros(len(ds.z.data),)
    elif model == 'harmonie':
        try:
            w = ds.tw.data
        except:
            w = np.zeros(len(ds.z.data),)
    elif model == 'g2s':
        w = 0.0*ds.z
    T = ds.t.data
    rho = ds.den.data
    P = ds.pres.data

    # Write out data
    if model == 'ecmwf' or 'harmonie':
        line = np.flipud(np.c_[z/1e3, u, v, w, T, rho/1e3, P/1e2])
    elif model == 'g2s':
        line = np.c_[z/1e3, u, v, w, T, rho/1e3, P/1e2]
    line_fmt = '%9.3f %8.3f %8.3f %8.3f %10.3f %11.4e %11.4e'
    #print(line)
    np.savetxt(fid, line, fmt=line_fmt, header=header, comments='#')
    return

def write_mean_profile(ds, fid, model='ecmwf'):
    """
    This routine writes the profile to a standard 1-D atmosphere
    format that can be used with NCPAprop.

    Parameters
    ----------
    ds : `xarray.Dataset`
        XArray Dataset containing atmosphere model

    fid : `str`
        Name of file to write to
    """
    print("*"*80)
    print("Writing 1-D profile to [ {} ]".format(fid))

    if model == 'ecmwf':
        z = ds.z.mean(axis=1).data
    elif model == 'g2s':
        z = ds.z.data
    u = ds.u.mean(axis=1).data
    v = ds.v.mean(axis=1).data
    try:
        w = ds.w.mean(axis=1).data
    except:
        w = 0.0*ds.hybrid
    T = ds.t.mean(axis=1).data
    rho = ds.den.mean(axis=1).data
    P = ds.pres.mean(axis=1).data

    # Write header
    header = (f'% 0, Z0, m, {min(z):.1f}\n'
              f'% 1, Z, km\n'
              f'% 2, U, m/s\n'
              f'% 3, V, m/s\n'
              f'% 4, W, m/s\n'
              f'% 5, T, degK\n'
              f'% 6, RHO, g/cm3\n'
              f'% 7, P, mbar')

    # Write out data
    if model == 'ecmwf':
        line = np.flipud(np.c_[z/1e3, u, v, w, T, rho/1e3, P/1e2])
    elif model == 'g2s':
        line = np.c_[z/1e3, u, v, w, T, rho/1e3, P/1e2]

    line_fmt = '%9.3f %8.3f %8.3f %8.3f %10.3f %11.4e %11.4e'
    np.savetxt(fid, line, fmt=line_fmt, header=header, comments='#')
    return
