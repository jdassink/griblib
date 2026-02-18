"""
Script to convert KNMI AWS data to NetCDF files
"""
import os
import sys
import glob
from argparse import ArgumentParser

import xarray as xr
from griblib import ECMWF
from griblib.utils.geo import get_great_circle_path
from griblib.io.ascii import write_profile

#I46RU = dict(lat=54.0, lon=84.8)

def main(argv):
    args = process_cli(argv)
    
    glob_key = f'{args.grib_dir}/{args.year}/*grib'
    grib_files = sorted(glob.glob(glob_key))

    for fid_grib in grib_files:
        try:
            prefix = fid_grib.split('/')[-1].split('.')[0]
            year = fid_grib.split('/')[-2]
            output_dir = f'{args.profile_dir}/{year}'
            if os.path.exists(output_dir) is False:
                os.makedirs(output_dir)
            fid_profile = f'{output_dir}/{prefix}.dat'

            print('-'*80)
            print('')
            print(f' -> Processing {fid_grib}')
            ds = process_ecmwf_grib(fid_grib)
            dsp = ds.interp(latitude=args.latitude,
                            longitude=args.longitude)

            print(f' -> Writing out {fid_profile}')
            write_profile(dsp, fid_profile)
            print('')
            print('-'*80)
            print('')

        except ValueError as e:
            print(e)
            pass

    return

def process_ecmwf_grib(fid_grib):
    my_atmos = ECMWF()
    request = [{'level_type': 'hybrid', 
                'var_list': ['z', 'lnsp', 't', 'u', 'v', 'w', 'q']
               }]

    ds = my_atmos.read_grib(fid_grib, request, compat='override')
    # Convert lnsp field to surface pressure (sp) and add to dataset
    sp = my_atmos.convert_lnsp_to_sp(ds.lnsp)
    ds = xr.merge([sp.to_dataset(), ds])
    ds = ds.drop(['lnsp'])

    # Compute altitude with respect to mean sea level
    (z, pres) = my_atmos.compute_altitude_pressure(t=ds.t,
                                                   q=ds.q,
                                                   ps=ds.sp,
                                                   z0=ds.z)

    # # Also compute height above ground (z = 0 km)
    # (hgt, _) = my_atmos.compute_altitude_pressure(t=ds.t,
    #                                               q=ds.q,
    #                                               ps=ds.sp, 
    #                                               z0=ds.z*0.0)
    # hgt = hgt.rename('hgt')
    # hgt.attrs= {'long_name': 'height above ground',
    #             'units' : 'm',
    #             'standard_name': 'height_above_ground'}
        
    dens = my_atmos.compute_density(pres, ds.t)

    ds = ds.drop(['z', 'sp'])
    #hgt.to_dataset(),
    ds = xr.merge([z.to_dataset(),
                   pres.to_dataset(),
                   dens.to_dataset(),
                   ds])
    return ds

def process_cli(argv):
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument('-gd', '--grib_dir',
        type=str, default='grib')

    # Required arguments
    parser.add_argument('-pd', '--profile_dir',
        type=str, default='profiles')

    # Required arguments
    parser.add_argument('-lon', '--latitude',
        type=float)

    # Required arguments
    parser.add_argument('-lat', '--longitude',
        type=float)

    # Optional arguments
    parser.add_argument('-y', '--year',
        type=str,
        default='*',
        metavar='Data year',
        )

    # # Optional arguments
    # parser.add_argument('-c', '--channels',
    #     nargs='?', type=str, metavar='SEED channel name',
    #      default='*'
    #     )
    parser.add_argument('-v', dest='verbose', action='store_true')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
   main(sys.argv[1:])