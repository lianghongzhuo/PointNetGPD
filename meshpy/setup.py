"""
Setup of meshpy python codebase
Author: Jeff Mahler
"""
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import os

class PostDevelopCmd(develop):
    def run(self):
        develop.run(self)

class PostInstallCmd(install):
    def run(self):
        install.run(self)

requirements = [
    'numpy',
    'scipy',
    'scikit-learn',
    'Pillow',
]

setup(name='meshpy',
    version='0.1.0',
    description='MeshPy project code',
    author='Matt Matl',
    author_email='mmatl@berkeley.edu',
    package_dir = {'': '.'},
    packages=['meshpy'],
    #ext_modules = [meshrender],
    install_requires=requirements,
    test_suite='test',
    cmdclass={
        'install': PostInstallCmd,
        'develop': PostDevelopCmd
    }
)
