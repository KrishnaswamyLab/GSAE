#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='GSAE',
      version='0.0.1',
      description='Graph Scattering Autoencoder',
      author='',
      author_email='',
      url='https://github.com/ec1340/GSAE',
      install_requires=[
            'pytorch-lightning'
      ],
      packages=find_packages()
      )

