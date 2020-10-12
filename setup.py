# -*- coding: utf-8 -*-
import os
from setuptools import setup
from setuptools import find_packages

# meta info
NAME = "pyctlops"
VERSION = "0.1"
AUTHOR = "Carlos Alberto da Costa Filho"
AUTHOR_EMAIL = "c.dacostaf@gmail.com"
URL = "https://github.com/cako/PyCurvelabLops"
DESCRIPTION = 'Thin PyLops wrapper for PyCurvelab'
LICENSE = "MIT"

if not os.path.exists('README.txt'):
    os.system("pandoc -o README.txt README.md")
LONG_DESCRIPTION = open('README.txt').read()


def main():
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,
        include_package_data=True,
        packages=find_packages(exclude=['pytests']),
        install_requires=[
            "pyct",
            "pylops",
            "numpy",
        ],
        license=LICENSE,
        test_suite='pytests',
        tests_require=['pytest'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Scientific/Engineering :: Mathematics'
        ],
        keywords='curvelet curvelab',
    )


if __name__ == '__main__':
    main()
