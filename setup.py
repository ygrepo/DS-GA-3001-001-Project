'''Setup file for the necessary packages to be installed.
'''
import codecs
import os
import re
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    required_libraries = f.read().splitlines()

setup(
    name="ts",
    version=find_version("ts", "__version__.py"),
    author="Yves Greatti",
    author_email="yg390@nyu.edu",
    description="M4 Forecasting Competition Comparison between n-beats and esrrn-gpu",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ygrepo/DS-GA-3001-001-Project.git",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="",
    install_requires=required_libraries,
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
    zip_safe=False,
)
