#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages

if sys.argv[-1] == 'setup.py':
    print "Please run 'python setup.py install' to install"
    print " "

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='SaCoRG',
    version='0.1.0',
    description='Sampling and Counting of Random Graphs with Prescribed Degree Sequence',
    long_description=readme,
    author='Abdulkadir Celikkanat',
    author_email='abdcelikkanat@gmail.com',
    url='https://github.com/abdcelikkanat/sacorg',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)