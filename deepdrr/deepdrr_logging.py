"""Define the logger for deepdrr.

The log level can be set from the command line by defining the environment variable DEEPDRR_LOG_LEVEL.
"""

import logging
import os
from colorlog import ColoredFormatter

logger = logging.getLogger("deepdrr")
ch = logging.StreamHandler()
ch.setLevel(os.environ.get("DEEPDRR_LOG_LEVEL", logging.INFO))
cf = ColoredFormatter(
    "%(asctime)s %(cyan)s%(name)-30s%(reset)s %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s")
ch.setFormatter(cf)
logger.addHandler(ch)
