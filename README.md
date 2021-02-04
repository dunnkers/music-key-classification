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
Once you have downloaded the file from [Here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files)
* Run `python src/data.py list -N <number_of_tracks` to create a list of tracks. Will fetch the corresponding 
audio_features objects from the spotify API in batches of 100, stores these objects in '<output_dir>/audio_features/'.
* Run `python src/data.py list --use-list <path_to_list>` to continue creating a list of tracks using the track list 
  object to which a path is provided
* Run `python src/data.py fetch` to fetch all the audio_analysis objects for the tracks in the track list in the output 
  directory.
* Run `python src/data.py <command> --help` to get more information on a command and its options.

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

Run `python src/data.py <command> --help` to get more information on a command and its options.

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

## Peregrine
First, upload the dataset. Do this using:
```shell
rsync -aP $PEREGRINE_USERNAME@peregrine.hpc.rug.nl:~/Key-Recognition/logs ./logs
```
... where `$PEREGRINE_USERNAME` is set as your Peregrine login name, e.g. your P- or S- number.

Login to Peregrine and submit a job using:

```shell
sbatch src/peregrine.sh
```

Download the log files using:
```shell
rsync -aP $PEREGRINE_USERNAME@peregrine.hpc.rug.nl:~/Key-Recognition/logs ./logs
```

Finally we can visualize the results using a Notebook. First, however, post-process the results, using:
```shell
sh src/peregrine_postprocess.sh $JOB_ID
```
...where `$JOB_ID` is the Peregrine job id you ran the analysis on.

Then, open up `src/peregrine_results.ipynb` and adjust the .csv filename to match. The results can now be visualized. ✨