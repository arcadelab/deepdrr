import logging
from rich.logging import RichHandler

def setup_log():
    log = logging.getLogger(__name__)
    ch = RichHandler(level=logging.NOTSET)
    log.addHandler(ch)

