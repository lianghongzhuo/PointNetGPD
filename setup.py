#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang

from setuptools import setup, find_packages

__version__ = "0.0.1"
setup(
    name="PointNetGPD",
    version=__version__,
    keywords=["grasping", "deep-learning"],
    description="Code for PointNetGPD",
    license="MIT License",
    url="https://github.com/lianghongzhuo/PointNetGPD",
    author="Hongzhuo Liang",
    author_email="liang@informatik.uni-hamburg.de",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[]
)
