# -*- coding: utf-8 -*-
"""
Compute geopotential and geometric altitudes on ECMWF model levels

.. module:: compute_geopotential_on_ecmwf_levels

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
import sys

def main(argv):
    my_atm = ECMWF()
    # get coefficients, hardcoded or from table file
    (pv, ps, T, q) = my_atm.ecmwf_L137_coefficients()
    #fid = '../data/ecmwf_L137.dat'
    #(pv, ps, T, q) = my_atm.ecmwf_L137_coefficients_from_file(fid)

    nlev = len(T)
    z_pot_gnd = 0.0

    # actual computation
    (ph, pf) = my_atm.compute_pressure_levels(pv, ps)

    (z_pot, z_geo) = my_atm.compute_geo_altitudes(ph, T, q, z_pot_gnd)
    dens = my_atm.compute_density(pf, T)

    # print out results
    for i in range(nlev,0,-1):
        k = nlev-i
        ml = k+1
        print ('{:03d} {:10.2f} {:10.2f} {:10.2f} {:12.4f} {:12.6f}'.format(
            ml, z_pot[k], z_geo[k], T.data[k], pf.data[k]/1e2, dens.data[k]))

if __name__ == "__main__":
   main(sys.argv[1:])