# -*- coding: utf-8 -*-
  
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='clipped_matrix_completion',
    version='0.1.0',
    description='',
    long_description='',
    author='',
    author_email='',
    install_requires=requirements,
    url='',
    license="",
    packages=find_packages(include=('clipped_matrix_completion')),
    test_suite=''
)

