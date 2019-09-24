#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ocvx",
    description="OCVX Project",
    author="Baptiste ESTEBAN, Charles GINANE, Danae MARMAI",
    author_email="baptiste.esteban@epita.fr, charles.ginane@epita.fr, danae.marmai@epita.fr",
    packages=required)
