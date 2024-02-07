from setuptools import setup, find_packages

setup(
    name='3d_mesh_neural_network',
    version='0.1.0',
    packages=find_packages(where='src'),  # Add 'where' argument if your packages are in 'src'
    package_dir={'': 'src'},  # Add this line to indicate the package directory
)