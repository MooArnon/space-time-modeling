##########
# Import #
##############################################################################

from setuptools import setup, find_packages
import os

############
# Variable #
##############################################################################

__version__ = 0.1
__author__ = "Arnon Phongsiang" 
__email__ = "arnon.phongsiang@gmail.com"
__github__ = "https://github.com/MooArnon"
__medium__ = "https://medium.com/@oomarnon"
__linkedin__ = "https://www.linkedin.com/in/arnon-pongsiang-320796214/"

packages: list[str] = find_packages(
    exclude=[
        'tests', 
        'tests.*', 
        'experiment',
        'experiment.*',
        'local*',
        'run_local*'
    ]
)

deep_learning_requirements = os.path.join('deep_learning_requirements.txt')

###############
# Description #
##############################################################################

with open("README.md", "r", encoding="utf-8") as fh:
    
    long_description = fh.read()

################
# Dependencies #
##############################################################################

with open("requirements.txt", "r") as file_obj:
    
    # reading the data from the file
    file_data = file_obj.read()

    # splitting the file data into lines
    dependencies = file_data.splitlines()
    
with open("deep_learning_requirements.txt", "r") as file_obj:
    deep_learning_dependencies = file_obj.read().splitlines()

#########
# Setup #
##############################################################################

setup(
    name='space_time_modeling',
    version=__version__,
    author=__author__,
    author_email=__email__,
    packages=packages,
    install_requires=dependencies,
    include_package_data=True,
    long_description = long_description,
    python_requires='>=3.9',
    extras_require={
        'deep': deep_learning_dependencies
    },
)

##############################################################################
