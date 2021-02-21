from setuptools import setup, find_packages

setup(name='exponential_gaussian_mixture',
      version='0.1',
      description='Fit an exponential and gaussian mixture to data in Python',
      url='https://github.com/fatmac78/gaussian-exponential-mixture',
      license='MIT',
      author='Ethan Homan',
      author_email='ethanwhoman@gmail.com',
      packages=find_packages(),
      install_requires=[
          'pandas>=0.24.0',
          'numpy',
          'scipy',
      ])
