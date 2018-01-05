import os
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'sklearn',
    'pandas',
    'numpy',
    'argparse',
    'scipy'
    ]

setup(name='SklearnTextClassify',
      version='0.1',
      description="Implement's basic text classification functionality via Sklearn" ,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False
      )