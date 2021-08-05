import logging
from rich.logging import RichHandler

log = logging.getLogger(__name__)
ch = RichHandler(level=logging.WARNING)
log.addHandler(ch)

