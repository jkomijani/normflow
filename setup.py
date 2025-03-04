# Copyright (c) 2021-2025 Javad Komijani

"""This is the setup script for `normflow`."""

from setuptools import setup


def readme():
    """Reads and returns the contents of the README.md file."""
    with open('README.md', encoding='utf-8') as f:
        return f.read()


packages = [
        'normflow',
        'normflow.action',
        'normflow.device',
        'normflow.lib',
        'normflow.lib.combo',
        'normflow.lib.indexing',
        'normflow.lib.linalg',
        'normflow.lib.matrix_handles',
        'normflow.lib.optim',
        'normflow.lib.spline',
        'normflow.lib.stats',
        'normflow.mask',
        'normflow.mcmc',
        'normflow.nn',
        'normflow.nn.gauge',
        'normflow.nn.matrix',
        'normflow.nn.scalar',
        'normflow.prior'
        ]

package_dir = {
        'normflow': 'src',
        'normflow.action': 'src/action',
        'normflow.device': 'src/device',
        'normflow.lib': 'src/lib',
        'normflow.lib.combo': 'src/lib/combo',
        'normflow.lib.indexing': 'src/lib/indexing',
        'normflow.lib.linalg': 'src/lib/linalg',
        'normflow.lib.matrix_handles': 'src/lib/matrix_handles',
        'normflow.lib.optim': 'src/lib/optim',
        'normflow.lib.spline': 'src/lib/spline',
        'normflow.lib.stats': 'src/lib/stats',
        'normflow.mask': 'src/mask',
        'normflow.mcmc': 'src/mcmc',
        'normflow.nn': 'src/nn',
        'normflow.nn.gauge': 'src/nn/gauge',
        'normflow.nn.matrix': 'src/nn/matrix',
        'normflow.nn.scalar': 'src/nn/scalar',
        'normflow.prior': 'src/prior'
        }

setup(
    name='normflow',
    version='2.0',
    description='Normalizing flow for generating lattice field configurations',
    packages=packages,
    package_dir=package_dir,
    url='http://github.com/jkomijani/normflow',
    author='Javad Komijani',
    author_email='jkomijani@gmail.com',
    license='MIT',
    zip_safe=False
)
