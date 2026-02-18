# -*- coding: utf-8 -*-
"""
AROME atmosphere model class

.. module:: arome

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
import xarray as xr
from copy import deepcopy

class AROME(Atmosphere):
    def __init__(self, cycle=None, **kwargs):
        Atmosphere.__init__(self)
        self.model = 'AROME'
        try:
            self.cycle = int(cycle)
        except Exception as e:
            print(e)
            self.cycle = ''
            pass

        return