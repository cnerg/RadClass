#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='RadClass',
      version='0.1.0',
      description='Semi-Supervised Machine Learning for Radiation Signature Identification',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Jordan Stomps',
      author_email='stomps@wisc.edu',
      url='https://github.com/CNERG/RadClass',
      packages=find_packages()
      )