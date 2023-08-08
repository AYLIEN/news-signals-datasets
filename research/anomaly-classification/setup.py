from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('VERSION') as f:
    version = f.read().strip()

setup(
    name="ac",
    version=version,
    requirements=requirements,
    packages=["ac"]
)
