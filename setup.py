#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='radclass',
      version='0.0.1',
      description='Radiation Isotropic Classifying Algorithm',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Jordan Stomps',
      author_email='stomps@wisc.edu',
      url='https://github.com/CNERG/MINOS.RadClass',
      packages=find_packages()
      )
