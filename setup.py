from setuptools import setup, find_packages

setup(
    name = 'cat_mod',
    version = '0.1.0',
    packages = find_packages(),
    install_requires = [
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.5.0',
        'tensorflow==2.16.1'
    ],
)