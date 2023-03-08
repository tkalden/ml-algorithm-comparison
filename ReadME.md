## Setting Up Local Environment

1. Download and install Miniconda from the official website: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Open a terminal window and create a new virtual environment:

```
conda create -n myenv python=3.10

```

Here, `myenv` is the name of your new virtual environment, and `python=3.10` specifies the version of Python you want to use. You can change these values to suit your needs.

1. Activate the virtual environment:

```
conda activate myenv

```

## How to install required packages using Conda

1. Once you have activated your virtual environment, you can use the following command to install the required packages:

```
conda install pandas jupyter seaborn scikit-learn keras tensorflow

```

This will install the latest version of each package that is compatible with Python 3.10.

## How to run the main file

```
python clothing_classifier_runner.py
python titanic_classifier_runner.py

