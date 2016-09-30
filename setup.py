from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()
    
setup(name='samplepy',
      version='1.0.1',
      description='sampling from univariate distributions',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Mathematics',
          ],
      keywords='sampling rejection importance Metropolis-Hastings',
      url='http://github.com/elena-sharova/samplepy',
      author='Elena Sharova',
      license='MIT',
      packages=['samplepy'],
      install_requires=[
          'numpy',
          'scipy'
          ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False)