from setuptools import setup


with open("VERSION") as f:
    version = f.read().strip()

# TODO: this is not the advised way to do this for a library
# but before fixing it, we need to know the best way to install spacy models
# and other data files that are not python modules.
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="news-signals",
    version=version,
    description="A library for working with text and timeseries data.",
    install_reuires=requirements,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    packages=["news_signals"],
    data_files=["LICENSE", "VERSION", "README.md"],
    include_package_data=True
)
