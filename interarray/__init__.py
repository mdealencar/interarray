# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

# author, version, license, and long description
__version__ = '0.0.1'
__author__ = 'Mauricio Souza de Alencar'

__doc__ = """
`interarray` implements extensions to the Esau-Williams heuristic for the
capacitaded minimum spanning tree (CMST) problem.

https://github.com/mdealencar/interarray
"""

__license__ = "LGPL-2.1-or-later"

import logging
logger = logging.getLogger('interarray')
logging.basicConfig(level=logging.WARNING)

info = logger.info
debug = logger.debug
warn = logger.warning
error = logger.error

# global module constants
MAX_TRIANGLE_ASPECT_RATIO = 50.
