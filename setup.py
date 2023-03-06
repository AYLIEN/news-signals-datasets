from setuptools import setup


with open("VERSION") as f:
    version = f.read().strip()


# Try not to add new dependencies unless they are really essential.
# If you add a new dependency, make sure to make the version range as wide as possible.
# TODO: we need to decide on the best way to install particular spacy models
# and other data files that are not python modules.
INSTALL_REQUIRES = [
    "appdirs>=1.4.4",
    "arrow>=1.1.1",
    "gdown>=4.6.4",
    "matplotlib>=3.5.3",
    "networkx<=2.7",
    "pandas>=1.4.1",
    "pyarrow>=11.0.0",
    "ratelimit>=2.2.1",
    "requests>=2.28.1",
    "scikit-learn>=1.1.0",
    "scipy>=1.8",
    "spacy>=3.4.1,<4.0.0",
    "sqlitedict>=1.7.0",
    "tqdm>=4.62.3",
    "Wikidata>=0.7.0"
]

setup(
    name="news-signals",
    version=version,
    description="A library for working with text and timeseries data.",
    install_requires=INSTALL_REQUIRES,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    packages=["news_signals"],
    data_files=["LICENSE", "VERSION", "README.md"],
    include_package_data=True
)
