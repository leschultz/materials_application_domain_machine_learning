import setuptools

# Package information
name = 'materials_application_domain'
version = '0.0.1'
description = 'Application domain of machine learning in materials science.'
url = 'https://github.com/leschultz/application_domain.git'
author = 'Lane E. Schultz'
author_email='laneenriqueschultz@gmail.com'
python_requires = '>=3.6'
classifiers = ['Programming Language :: Python :: 3',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               ]
package_dir = {'': 'mad'}
packages=setuptools.find_packages(where='mad')

# Passing variables to setup
setuptools.setup(
                 name=name,
                 version=version,
                 description=description,
                 url=url,
                 author=author,
                 author_email=author_email,
                 package_dir=package_dir,
                 packages=packages,
                 python_requires=python_requires,
                 classifiers=classifiers,
                 )
