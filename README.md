# py-rogue-detection
A python repository for detection of corn rogues in video and imagery.

## Current Modules

### analysis
Houses the statistical modeling that is performed on ground rogues data. See the README in the analysis folder for more information.

### data_augmentation
Houses the data augmentation code that is used to create training data for the deep learning models. See the README in the data_augmentation folder for more information.

### ddb_tracking
Houses the code that is used to track rogues data in DynamoDB. See the README in the ddb_tracking folder for more information.

### ground_data_processing
Processing for ground data, most commonly video imagery collected on a hagie rig with 3 camera views. See the README in the ground_data_processing folder for more information.

## Installation 

### Poetry
1) [Set up SSH](https://github.com/SenteraLLC/install-instructions/blob/master/ssh_setup.md)
2) Install [pyenv](https://github.com/SenteraLLC/install-instructions/blob/master/pyenv.md) and [poetry](https://python-poetry.org/docs/#installation)
3) Install package

        >> git clone git@github.com:SenteraLLC/py-rogue-detection.git
        >> cd py-rogue-detection
        >> pyenv install $(cat .python-version)
        >> poetry install
        
4) Set up ``pre-commit`` to ensure all commits to adhere to **black** and **PEP8** style conventions.

        >> poetry run pre-commit install
        
### Conda
It is recommended you install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) over Anaconda for a quicker install time and to save space on unneeded packages. Make sure you install Miniconda for Python 3.X.

1) [Set up SSH](https://github.com/SenteraLLC/install-instructions/blob/master/ssh_setup.md)
2) Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
3) Install package

        >> git clone git@github.com:SenteraLLC/py-rogue-detection.git
        >> cd py-rogue-detection
        >> conda env create -f environment.yml
        >> conda activate rogue-venv
        >> pip install -e .
        
4) Within the conda shell, set up ``pre-commit`` to ensure all commits to adhere to **black** and **PEP8** style conventions.

        >> pre-commit install
# py-rogue-copy
