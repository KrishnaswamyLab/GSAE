#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='GSAE',
      version='0.0.1',
      description='Graph Scattering Autoencoder',
      author='Egbert Castro',
      author_email='egbert.castro@yale.edu',
      url='https://github.com/ec1340/GSAE',
      install_requires=[
            'pytorch-lightning',
            'torch'
      ],
      packages=find_packages()
      )

