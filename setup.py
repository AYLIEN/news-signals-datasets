from setuptools import setup


with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="news-signals",
    version=version,
    description="A library for working with text and timeseries data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    packages=["news_signals"],
    package_data=["resources/test"],
    data_files=["LICENSE", "VERSION", "README.md"],
    include_package_data=True
)
