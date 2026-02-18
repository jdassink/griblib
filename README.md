*******
GRIBLIB
*******

A versatile Python library for working with GRIB files in Python, combining the power of `eccodes`, `cfgrib`, `pyproj` and `xarray`.

- 3D GRIB file analysis
- Geopotential/geometrical altitude computation (useful for ECMWF files)
- Horizontal transects
- Vertical profiles and transects
- Infrasound parameters of interest (adiabatic and effective sound speed maps) 
- Gravity wave parameters of interest (Brunt-Vaisala frequency, Richardson number)
- Micropressure turbulence analysis
- Example scripts and Jupyter Notebooks


Installing this branch
----------------------

It is best if the required packages are first installed with
`conda`:

.. code-block:: console

    conda install numpy xarray cfgrib pyproj pandas xarray


Clone:

.. code-block:: console

    git clone git@gitlab.com:KNMI/RDSA/research/griblib.git


Install:

.. code-block:: console

    cd griblib
    pip install -e .


Example Workflow
----------------
