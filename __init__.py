import logging
import os


def get_logger(name):
    """ Get a logger. The results are stored in out/logs. """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not os.path.exists("out/logs"):
        os.makedirs("out/logs")
    fh = logging.FileHandler(f"out/logs/{name}.log")

    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
