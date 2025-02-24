import sys
import logging


def create_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
