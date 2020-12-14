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

## Running the "naive" method on the dataset

To analyse the downloaded audio analysis using a naive method, use the `src/naive.py` script:

```
python src/naive.py --data-dir dataset --csv output.csv
```

Make sure the `data-dir` matches the `output-dir` from the fetching step.

