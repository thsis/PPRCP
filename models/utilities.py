"""
Basic utilities like:

* getting a logger
"""
import logging


def get_logger(filepath):
    """Create a logger with a fixed file.

    Parameters
    ----------

    * `filepath`: `str`
        path and name of log file

    Returns
    -------

    * `logger`:
        Logger from the `logging` module.

    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filepath)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
