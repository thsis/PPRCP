"""
Setup for package
"""

import os
from setuptools import setup, find_packages


def read(fname):
    """Return contents of a file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name="PPRCP",
      author="Thomas Siskos",
      author_email="thomas.siskos91@gmail.com",
      description="Models and algorithms for my Master thesis",
      license="MIT",
      url="https://github.com/thsis/PRPP",
      packages=find_packages(),
      long_description=read("README.md"))
