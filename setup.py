#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup experimaestro IR."""

from pathlib import Path
import os
import re
from setuptools import setup, find_packages


def get_description():
    """Get long description."""

    with open("README.md", "r") as f:
        desc = f.read()
    return desc


basepath = Path(__file__).parent
install_requires = (basepath / "requirements.txt").read_text()
install_requires = re.sub(
    r"^(git\+https.*)egg=([_\w-]+)$", r"\2@\1", install_requires, 0, re.MULTILINE
)

# Removes dependencies if building documentation
if os.environ.get("DOC_BUILDING", 0) == "1":
    install_requires = re.sub(r"SKIP_DOCBUILD.*", r"", install_requires, 0, re.DOTALL)

setup(
    name="experimaestro-ir",
    python_requires=">=3.7",
    description="Experimaestro common module for IR experiments",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    author="Benjamin Piwowarski",
    author_email="benjamin@piwowarski.fr",
    url="https://github.com/bpiwowar/experimaestro-ir",
    packages=find_packages(exclude=["test*", "tools"]),
    use_scm_version=True,
    license="GPL 3.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=install_requires,
    extras_require={"neural": ["torch>=1.7", "tensorboard"]},
    setup_requires=["setuptools_scm", "setuptools >=30.3.0"],
    entry_points={
        "datamaestro.repositories": {
            "ir = xpmir:Repository",
            "irds=xpmir.datasets.irds:Repository",
        }
    },
)
