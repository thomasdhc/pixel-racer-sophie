from setuptools import setup, find_packages

setup(name='racer',
      version='1.0.0',
      description='Core racer code',
      long_description='',
      url='',
      author='Thomas (Dong Hyuk) Chang',
      author_email='thomaschang26@gmail.com',
      license='',
      packages=find_packages(),
      include_package_data=False,
      install_requires=[
          'numpy==1.14.1',
          'pylint==1.8.2',
          'matplotlib==2.1.2',
          'fire==0.1.0',
          'tensorflow==1.6.0',
      ],
      zipsafe=False)
