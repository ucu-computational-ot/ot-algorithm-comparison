import logging
import sys


def setup_logger(
        name: str = "uot",
        level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger


logger = setup_logger()
