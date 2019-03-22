from setuptools import (
    setup,
    find_packages,
)

__version__ = '0.1.0'

setup(
    name='pyboretum',
    version=__version__,
    description="Code to build and experiment with custom decision trees",
    author='Vincent Stigliani',
    author_email='vincent@picwell.com',
    url='http://picwell.com',
    packages=find_packages(exclude=[
        'tests',
        '*.tests',
        'figures',
        'miscellaneous_natebooks',
        'utils',
    ]),
    setup_requires=[
        'pytest-runner'
    ],
    install_requires=[
        'sortedcontainers',
        'pandas',  # TODO; limit this to numpy eventually?
    ],
    tests_require=[
        'pytest',
    ],
)
