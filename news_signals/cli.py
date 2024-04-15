import sys
import subprocess
import logging

from news_signals.log import create_logger


logger = create_logger(__name__, level=logging.INFO)


def generate_dataset():
    logger.info("Generating dataset via cli command `generate-dataset`")
    print(f"Args: {sys.argv[1:]}")
    sys.exit(subprocess.call([sys.executable, 'bin/generate_dataset.py'] + sys.argv[1:]))
