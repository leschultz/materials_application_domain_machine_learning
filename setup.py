import setuptools

# Package information
name = 'madml'
version = '0.3.9'  # Need to increment every time to push to PyPI
description = 'Application domain of machine learning in materials science.'
url = 'https://github.com/leschultz/'\
      'materials_application_domain_machine_learning.git'
author = 'Lane E. Schultz'
author_email = 'laneenriqueschultz@gmail.com'
python_requires = '>=3.6'
classifiers = ['Programming Language :: Python :: 3',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               ]
packages = setuptools.find_packages(where='src')
install_requires = [
                    'matplotlib',
                    'scipy',
                    'scikit-learn',
                    'pandas',
                    'numpy',
                    'pathos',
                    'tqdm',
                    'pytest',
                    'openpyxl',
                    'docker',
                    'tensorflow',
                    ]

long_description = open('README.md').read()

# Passing variables to setup
setuptools.setup(
                 name=name,
                 version=version,
                 description=description,
                 url=url,
                 author=author,
                 author_email=author_email,
                 packages=packages,
                 package_dir={'': 'src'},
                 package_data={'madml': ['data/*', 'templates/docker/*']},
                 python_requires=python_requires,
                 classifiers=classifiers,
                 install_requires=install_requires,
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 )
