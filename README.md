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

* Run `python src/data.py list -N <number_of_tracks` to create a list of tracks. Will fetch the corresponding 
audio_features objects from the spotify API in batches of 100, stores these objects in '<output_dir>/audio_features/'.
* Run `python src/data.py list --use-list <path_to_list>` to continue creating a list of tracks using the track list 
  object to which a path is provided
* Run `python src/data.py fetch` to fetch all the audio_analysis objects for the tracks in the track list in the output 
  directory.
* Run `python src/data.py <command> --help` to get more information on a command and its options.

## Running the "naive" method on the dataset

To analyse the downloaded audio analysis using a naive method, use the `src/naive.py` script:

```
python src/naive.py --data-dir dataset --csv output.csv
```

Make sure the `data-dir` matches the `output-dir` from the fetching step. For example, running the command above after fetching 250 tracks, the result is:

```
[[14.  0.  0.  0.  1.  0.  0.  3.  0.  0.  0.  0.]
 [ 0. 13.  1.  0.  0.  0.  6.  0.  0.  0.  0.  1.]
 [ 0.  0. 16.  0.  0.  0.  0.  1.  0.  1.  0.  0.]
 [ 1.  1.  0.  1.  0.  0.  0.  0.  0.  0.  2.  0.]
 [ 0.  0.  0.  0. 16.  0.  0.  0.  0.  5.  0.  0.]
 [ 1.  0.  1.  0.  0.  6.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  4.  0.  0.  0.  0.  0.]
 [ 2.  0.  1.  0.  0.  0.  0. 11.  0.  0.  0.  0.]
 [ 0.  5.  2.  1.  0.  3.  1.  1. 10.  0.  1.  0.]
 [ 0.  0.  2.  0.  0.  0.  0.  0.  0. 11.  0.  0.]
 [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  5.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  6.]]
N=158
Overall accuracy: 71.52%
```

Note that tracks that are in minor key are ignored for now, so N is lower than 250.

