#!/usr/bin/env python

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'lymphoma-subtype-classification'
DESCRIPTION = 'A short description of the project'
URL = 'https://github.com/mikelemilia/lymphoma-subtype-classification.git'
EMAIL = 'mikele.milia@studenti.unipd.it'
LICENSE = 'MIT'
AUTHOR = 'Mikele Milia'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    'keras==2.4.3',
    'tensorflow==2.4.0',
    'tensorflow-datasets',
    'opencv-python==3.4.1.15',
    'opencv-contrib-python==3.4.1.15',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=[
            "tests",
            "*.tests",
            "*.tests.*",
            "tests.*"
        ]
    ),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['scripts'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)
