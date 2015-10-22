from setuptools import setup, find_packages


setup(name='kernel_exp_family',
      version='0.1',
      description='Various estimators for the infinite dimensional exponential family model',
      url='https://github.com/karlnapf/kernel_exp_family',
      author='Heiko Strathmann',
      author_email='heiko.strathmann@gmail.com',
      license='BSD3',
      packages=find_packages('.'),
      zip_safe=False)
