#!/usr/bin/env python

"""
setup.py - Setup script for swmm-pandas python package
Author:    See AUTHORS

"""

from setuptools import find_packages, setup

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

with open("README.md") as readme_file:
    readme = readme_file.read()


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("requirements.txt", session=False)
# reqs is a list of requirement
requirements = [str(ir.requirement) for ir in install_reqs]

# find packages and prepend with swmm
packages = [f"swmm.{package}" for package in find_packages(where="./src/swmm")]

test_requirements = [
    "pytest>=3",
]

setup(
    name="swmm-pandas",
    version="0.1.0",
    packages=packages,
    package_dir={"": "src"},
    zip_safe=False,
    install_requires=requirements,
    description="SWMM binary outputfile reader and API",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/OpenWaterAnalytics/swmm-python",
    author="See README.md",
    maintainer_email="ckaros@outlook.com",
    license="CC BY 4.0",
    python_requires=">=3.6",
    keywords="swmm5, swmm, stormwater, hydraulics, hydrology, ",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 4 - Beta",
    ],
    include_package_data=True,
    test_suite="tests",
    tests_require=test_requirements,
)
