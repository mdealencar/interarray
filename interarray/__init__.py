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

import sys
import logging

logger = logging.getLogger(__name__)

# WARNING, ERROR, CRITICAL go to stderr
stderr_handler = logging.StreamHandler()
stderr_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(levelname)s: %(message)s '
                              '[%(name)s:%(funcName)s]')
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)

# DEBUG, INFO go to stdout (as well as any level below WARNING)
def _log_stdout_filter(record):
    return record.levelno < logging.WARNING

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(_log_stdout_filter)
stdout_handler.setLevel(logging.NOTSET)
logger.addHandler(stdout_handler)

info = logger.info
debug = logger.debug
warn = logger.warning
error = logger.error

# global module constants
MAX_TRIANGLE_ASPECT_RATIO = 50.
