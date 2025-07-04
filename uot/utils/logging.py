import logging
import sys


def setup_logger(
        name: str = "uot",
        level: int = logging.INFO,
        stream=sys.stdout
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler(stream)
        fmt = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger


logger = setup_logger()
