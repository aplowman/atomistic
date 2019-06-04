"""Pip installation script for `atomistic`."""

import os
import re
from setuptools import find_packages, setup

package_name = 'atomistic'


def get_version():

    ver_file = '{}/_version.py'.format(package_name)
    with open(ver_file) as handle:
        ver_str_line = handle.read()

    ver_pattern = r'^__version__ = [\'"]([^\'"]*)[\'"]'
    match = re.search(ver_pattern, ver_str_line, re.M)
    if match:
        ver_str = match.group(1)
    else:
        msg = 'Unable to find version string in "{}"'.format(ver_file)
        raise RuntimeError(msg)

    return ver_str


def get_long_description():

    readme_file = 'README.md'
    with open(readme_file, encoding='utf-8') as handle:
        contents = handle.read()

    return contents


package_data = [
    os.path.join(*os.path.join(root, f).split(os.path.sep)[1:])
    for root, dirs, files in os.walk(os.path.join(package_name, 'data'))
    for f in files
]

setup(
    name=package_name,
    version=get_version(),
    description=('Build atomistic structures such as grain boundaries with '
                 'Python'),
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Adam J. Plowman',
    author_email='adam.plowman@manchester.ac.uk',
    packages=find_packages(),
    package_data={
        package_name: package_data,
    },
    install_requires=[
        'numpy',
        'plotly',
        'vecmaths',
        'bravais',
        'mendeleev',
        'spglib',
        'beautifultable',
    ],
    project_urls={
        'Github': 'https://github.com/aplowman/atomistic',
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: OS Independent',
    ],
)
