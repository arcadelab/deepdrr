# https://johnpaton.net/posts/redirect-logging/
import logging
import contextlib

class OutputLogger:
    def __init__(self, name="root", level="INFO"):
        self.logger = logging.getLogger(name)
        self.name = self.logger.name
        self.level = getattr(logging, level)

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self): pass