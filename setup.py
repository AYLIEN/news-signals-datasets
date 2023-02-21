from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="news_signals",
    version=version,
    packages=["news_signals"],
    install_requires=requirements
)
