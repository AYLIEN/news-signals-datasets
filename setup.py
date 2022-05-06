from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="aylien_ts_datasets",
    version=version,
    packages=["aylien_ts_datasets"],
    install_requires=requirements
)
