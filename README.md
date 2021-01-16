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

(run `python -m venv venv` for older versions)

## Fetching the dataset

Run `python src/data.py fetch <N>` to fetch N objects from the spotify API and put it in a dataset. Example:

```shell
python src/data.py fetch 100 -o dataset
```

To fetch 100 track analyses. Will grab the first found tracks from some playlist from '[The Million Playlist Dataset](https://www.kaggle.com/sadakathussainfahad/spotify-million-playlist-dataset)' stored in the `./spotify_million_playlist_dataset/data` folder. So, make sure to have at least the amount of tracks in the playlist files that you want to fetch.

Run `python src/data.py resume` to resume fetching the objects from the spotify API if the fetching was interrupted. Assumes a valid meta file is present in the output directory.

Run `python src/data.py check` to check if the data in OUTPUT_DIR is complete and valid.

Run `python src/data.py <command> --help` to get more information on a command and its options.

## Running the models

To run one of the models, use the `src/key_recognition.py` script.

```
python src/key_recognition.py -h
usage: key_recognition.py [-h] [--data-dir DATA_DIR] [--give-mode]
                          [--test-split TEST_SPLIT] [--csv CSV] [--table]
                          {naive,hmm} ...

positional arguments:
  {naive,hmm}
    naive               Test classification using the naive method.
    hmm                 Test classification using the HMM method.

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   The directory where the track data is stored to use
                        for the analysis.
  --give-mode           Optionally test the model with given mode
                        (major/minor).
  --test-split TEST_SPLIT
                        The fraction of samples to use as testing data
  --csv CSV             Optional filename of a CSV file to store the resulting
                        confusion matrix
  --table               Whether or not to print a table of all the test
                        samples and their classification
```

For example the naive method (it's highly recommended _not_ to train this model):

```
python src/key_recognition.py --give-mode naive --no-training
Collecting training data...
Collecting testing data...
Data collected.
Testing model...
Done.
Overall error: 34.00%
```

And the HMM method:

```
python src/key_recognition.py --give-mode hmm
Collecting training data...
Collecting testing data...
Data collected.
Formatting training data...
Done.
Training minor model...
Trained minor model. Converged: True
Training major model...
Trained major model. Converged: True
Done.
Copying models...
Done
Testing model...
Done.
Overall error: 31.20%
```

