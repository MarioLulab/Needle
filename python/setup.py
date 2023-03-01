"""Setup the package."""
import os
import shutil
import sys
import sysconfig
import platform

from setuptools import find_packages
from setuptools.dist import Distribution


from setuptools import setup


__version__ = "0.1.0"


setup(
    name="needle",
    version=__version__,
    description="CMU-DLsys",
    zip_safe=False,
    packages=find_packages(),
    package_dir={"needle": "needle"},
    url="dlsyscourse.org"
)