from setuptools import setup


with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="news_signals",
    version=version,
    packages=["news_signals"]
)
