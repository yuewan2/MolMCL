import os
from setuptools import setup, find_packages

long_description = '''Repository for Molecule Multi-channel Learning'''

setup(
    name='molmcl',
    version='0.0.1',
    author='Yue Wan',
    author_email='yuw253@pitt.edu',
    py_modules=['molmcl'],
    description='Molecule Multi-channel Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages()
)
