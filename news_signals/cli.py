import sys
import subprocess
import logging
import os
from pathlib import Path

from news_signals.log import create_logger


logger = create_logger(__name__, level=logging.INFO)


path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))


def generate_dataset():
    logger.info("Generating dataset via cli command `generate-dataset`")
    logger.info(f"Args: {sys.argv[1:]}")
    sys.exit(subprocess.call([sys.executable, str(path_to_file / 'generate_dataset.py')] + sys.argv[1:]))
