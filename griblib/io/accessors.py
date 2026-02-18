# -*- coding: utf-8 -*-
"""
Module providing the xarray Dataset and DataArray accessors for reading, 
writing, and visualizing array processing results.

.. module:: accessors

:author:
    Jelle Assink (jelle.assink@knmi.nl)

:copyright:
    2020, Jelle Assink

:license:
    This code is distributed under the terms of the
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from xarray import (Dataset, open_dataset, register_dataset_accessor,
                    register_dataarray_accessor)
#from xarray.plot.utils import label_from_attrs
from griblib.utils.plotting import plot_pressure
from pandas import to_datetime

from functools import wraps

# @register_dataarray_accessor('_get_plot_label')
# class LabelAccessor():
#     def __init__(self, xarrayDataArray):
#         self._obj = xarrayDataArray

#     def __call__(self):
#         """
#         Get a string which can be used to label plots.
#         """
#         label = list(self._obj.standard_name.replace('_', ' '))
#         label = label[0].upper() + ''.join(label[1:])
#         try:
#             label += ', {}'.format(self._obj.plot_units)
#         except AttributeError:
#             pass
#         return label

@register_dataset_accessor('plot_pressure')
class PlotAccessor():
    def __init__(self, xarrayDataset):
        self._obj = xarrayDataset

    @wraps(plot_pressure)
    def __call__(self, *args, **kwargs):
        """
        Plot pressure fiel results using the
        :func:`~griblib.utils.plotting.plot_pressure` function.

        The docstring and signature are taken from
        :func:`~griblib.utils.plotting.plot_pressure`. Do not edit
        here.
        """
        return plot_pressure(self._obj, *args, **kwargs)
