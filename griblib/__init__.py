#import requests
#import warnings

from griblib.io import accessors
from griblib.models.ecmwf import ECMWF
from griblib.models.harmonie import HARMONIE
from griblib.models.arome import AROME

__all__ = ["ECMWF", "HARMONIE", "AROME"]

# Version
try:
    # - Released versions just tags:       1.10.0
    # - GitHub commits add .dev#+hash:     1.10.1.dev3+g973038c
    # - Uncom. changes add timestamp: 1.10.1.dev3+g973038c.d20191022
    from griblib.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version.
    # We could throw a warning here, but this case *should* be
    # rare. empymod should be installed properly!
    from datetime import datetime
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')

# if __name__ == '__main__':
#     import doctest
#     doctest.testmod(exclude_empty=True)
