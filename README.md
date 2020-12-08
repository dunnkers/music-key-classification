# Machine Learning Project: Musical key recognition using a Hidden Markov Model (HMM)
This is the repo for managing the code of the machine learning pipeline for our group project.

## Installation
Assuming you use Python 3.8, Virtualenv 20.2.1 and 

Run:

```shell
 python -m virtualenv venv
 . venv/bin/activate
 pip install -r requirements.txt
```

## Fetching the dataset

Run `python src/data.py fetch N` to fetch N objects from the spotify API and put it in a dataset.

Run `python src/data.py resume` to resume fetching the objects from the spotify API if the fetching was interrupted. Assumes a valid meta file is present in the output directory.

Run `python src/data.py check` to check the dataset.

Run `python src/data.py <command> --help` to get more information on a command and its options.