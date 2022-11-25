#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup experimaestro IR."""

from pathlib import Path
import os
import re
from setuptools import setup

# Handles dependencies here
basepath = Path(__file__).parent
install_requires = (basepath / "requirements.txt").read_text()
install_requires = re.sub(
    r"^(git\+https.*)egg=([_\w-]+)$", r"\2@\1", install_requires, 0, re.MULTILINE
)

if os.environ.get("DOC_BUILDING", 0) == "1":
    install_requires = re.sub(r"SKIP_DOCBUILD.*", r"", install_requires, 0, re.DOTALL)

setup(
    install_requires=install_requires,
)
