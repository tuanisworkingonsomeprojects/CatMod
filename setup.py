from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    description = f.read()


setup(
    name = 'cat_mod',
    version = '0.5.0',
    packages = find_packages(),
    install_requires = [
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.5.0',
        'tensorflow==2.16.1'
    ],
    long_description = description,
    long_description_content_type = 'text/markdown',
)