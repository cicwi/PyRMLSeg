#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import os.path

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

with open(os.path.join('rmlseg','VERSION')) as version_file:
    version = version_file.read().strip()

requirements = [
     'astra-toolbox',
     'numpy',
     'matplotlib',
     'scipy',
]

setup_requirements = [ ]

test_requirements = [ ]

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'watchdog',
    'coverage',

    ]

setup(
    author="Henri DER SARKISSIAN, Nicola VIGANÒ",
    author_email='N.R.Vigano@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License (MIT)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Relaxed multi-levelset segmentation package.",
    install_requires=requirements,
    license="MIT License",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rmlseg',
    name='rmlseg',
    packages=find_packages(include=['rmlseg']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={ 'dev': dev_requirements },
    url='https://github.com/cicwi/rmlseg',
    version=version,
    zip_safe=False,
)
