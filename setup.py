#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup experimaestro IR."""

from setuptools import setup, find_packages
import os
import imp

def get_description():
    """Get long description."""

    with open("README.md", 'r') as f:
        desc = f.read()
    return desc


DEVSTATUS = "alpha"
VERSION = "0.0.1"

setup(
    name='experimaestro_ir',
    version=VERSION,
    python_requires=">=3.5",
    description='Experimaestro common module for IR experiments',
    long_description=get_description(),
    long_description_content_type='text/markdown',
    author='Benjamin Piwowarski',
    author_email='benjamin@piwowarski.fr',
    url='https://github.com/bpiwowar/experimaestro-ir',
    packages=find_packages(exclude=['test*', 'tools']),
    install_requires=[ 
        "experimaestro>=0.5.7",
        "datamaestro_text",
        "pyserini>=0.7.0",
        "capreolus",
        "pytrec_eval"
    ],
    license='GPL 3.0',
    classifiers=[
        'Development Status :: %s' % DEVSTATUS,
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
