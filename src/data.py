import operator
from argparse import ArgumentParser
from functools import partial, reduce
from math import ceil
from os import getcwd, makedirs
from os.path import exists, isdir, join
from pathlib import Path
from pickle import dump, load
from shutil import rmtree
from typing import Generator

from dill import dump, load
from halo import Halo

from args import get_args
from constants import AUDIO_ANALYSIS, AUDIO_FEATURES
from meta import Meta
from mpl import list_track_ids, track_id_generator
from process import extract_audio_features, extract_track_analysis
from track_analysis import n_track_analyses_generator
from track_features import n_track_features
from tracklist import TrackList


def get_data_dir(output_dir, data_type) -> str:
    return join(output_dir, data_type)


def get_track_data_path(output_dir, data_type, track_id) -> str:
    return join(get_data_dir(output_dir, data_type), track_id) + '.pickle'


def get_audio_analysis_path(output_dir, track_id) -> str:
    return get_track_data_path(output_dir, AUDIO_ANALYSIS, track_id)

def load_analysis(output_dir, track_id):
    with open(get_track_path(output_dir, track_id), 'rb') as f:
        return load(f)


def get_audio_features_path(output_dir, track_id) -> str:
    return get_track_data_path(output_dir, AUDIO_FEATURES, track_id)


def start_fetching(mpl_dir, output_dir, n_tracks):
    if not exists(mpl_dir) or not isdir(mpl_dir):
        print(f"{mpl_dir} is not an existing directory")
        exit()
    if exists(output_dir) and isdir(output_dir):
        print(f"{output_dir} is an existing directory, deleting it")
        rmtree(output_dir)
    makedirs(output_dir)
    print("starting fetching...")
    track_ids = list_track_ids(mpl_dir, n_tracks)
    meta = create_meta(track_ids)
    meta.dump(output_dir)
    track_analyses = n_track_analyses_generator(track_ids)
    for track_analysis in track_analyses:
        if 'track_not_found' in track_analysis:
            meta.remove_track_id(track_analysis['track_not_found'])
            print(f"removed {track_analysis['track_not_found']} from dataset")
            continue
        extracted = extract_track_analysis(track_analysis)
        store_extracted_analysis(output_dir, extracted)
    print('done fetching')
    meta.dump(output_dir)

def store_datapoint(output_dir, data_type, datapoint) -> None:
    with open(get_track_data_path(output_dir, data_type, datapoint['id']), 'wb') as f:
        dump(datapoint, f)


def store_extracted_features(output_dir, extracted_audio_features) -> None:
    store_datapoint(output_dir, AUDIO_FEATURES, extracted_audio_features)


def store_extracted_analysis(output_dir, extracted_track_analysis) -> None:
    store_datapoint(output_dir, AUDIO_ANALYSIS, extracted_track_analysis)


def load_datapoint(output_dir, data_type, track_id) -> dict:
    with open(get_track_data_path(output_dir, data_type, track_id), 'rb') as f:
        return load(f)


def have_datapoint(output_dir, data_type, track_id) -> bool:
    return exists(get_track_data_path(output_dir, data_type, track_id))


def get_datapoint_ids(output_dir, data_type) -> Generator:
    path_gen = Path(get_data_dir(output_dir, data_type)).glob('*.pickle')

    def gen():
        for path in path_gen:
            yield path.stem
    return gen()


def count_data_points(output_dir, data_type):
    return len(list(get_datapoint_ids(output_dir, data_type)))


def load_features(output_dir, track_id) -> dict:
    return load_datapoint(output_dir, AUDIO_FEATURES, track_id)


def load_analysis(output_dir, track_id) -> dict:
    return load_datapoint(output_dir, AUDIO_ANALYSIS, track_id)


def create_track_list(track_ids) -> TrackList:
    _track_list = TrackList()
    _track_list.set_track_ids(track_ids)
    return _track_list


def get_n_track_ids(track_id_gen, n, have_id):
    track_ids = []
    next_id = next(track_id_gen, None)
    while next_id is not None and len(track_ids) < n:
        if not have_id(next_id):
            track_ids.append(next_id)
        next_id = next(track_id_gen, None)
    return track_ids


def listing_track_id_generator(output_dir, mpl_data_path, have_track_id):
    missing_tids = get_missing(output_dir, AUDIO_FEATURES)
    for tid in missing_tids:
        yield tid
    mpl_tid_generator = track_id_generator(mpl_data_path, have_track_id)
    for tid in mpl_tid_generator:
        yield tid


def list_tracks(mpl_data_path, output_dir, n, list_dir='', _track_list: TrackList = None) -> None:
    spinner = Halo(text='Listing tracks', spinner='dots')
    spinner.start()
    if not exists(get_data_dir(output_dir, AUDIO_FEATURES)):
        makedirs(get_data_dir(output_dir, AUDIO_FEATURES))
    # We will count the amount of tracks per key and mode
    key_counts = dict()
    required_per_key = ceil(n / 24)
    for key in range(24):
        key_counts[key] = 0
    # Start from a provided track list or create a new one
    track_list_complete = False
    if _track_list is not None:
        track_list_complete = True
        track_id_gen = listing_track_id_generator(output_dir, mpl_data_path, _track_list.have_track_id)
        for id in get_datapoint_ids(output_dir, AUDIO_FEATURES):
            f = load_features(output_dir, id)
            key = f['key'] + (f['mode'] * 12)
            key_counts[key] += 1
    else:
        track_id_gen = track_id_generator(mpl_data_path)
        _track_list = TrackList()
        _track_list.set_desired_tracks_amount(n)
    total = count_data_points(output_dir, AUDIO_FEATURES)
    _track_list.dump(output_dir)

    def finished():
        return reduce(operator.and_, [key_counts[key] >= required_per_key for key in range(24)]) or \
               count_data_points(output_dir, AUDIO_FEATURES) >= n
    while not finished():
        have_track = partial(have_datapoint, output_dir, AUDIO_FEATURES)
        track_ids = get_n_track_ids(track_id_gen, 100, have_track)
        track_feats = n_track_features(track_ids)
        for track_feat in track_feats:
            extracted_track_features = extract_audio_features(track_feat)
            key = extracted_track_features['key'] + (extracted_track_features['mode'] * 12)
            if key_counts[key] < required_per_key:
                key_counts[key] += 1
                if not track_list_complete:
                    _track_list.add_track_id(extracted_track_features['id'])
                store_extracted_features(output_dir, extracted_track_features)
                total += 1
            _track_list.dump(output_dir)
            if finished():
                break
        perc = 100 * (total / n)
        spinner.start(f'Listing tracks ({perc:.2f}%)')
    spinner.stop()
    _track_list.dump(output_dir)
    if list_dir:
        _track_list.dump(list_dir)


def fetch(output_dir):
    spinner = Halo('Fetching tracks', spinner='dots')
    spinner.start()
    _track_list = TrackList.load_from_dir(output_dir)
    track_ids = get_missing(output_dir, AUDIO_ANALYSIS)
    if not exists(get_data_dir(output_dir, AUDIO_ANALYSIS)):
        makedirs(get_data_dir(output_dir, AUDIO_ANALYSIS))
    track_analyses = n_track_analyses_generator(track_ids)
    count = 0
    for track_analysis in track_analyses:
        if 'track_not_found' in track_analysis:
            _track_list.remove_track_id(track_analysis['track_not_found'])
            print(f"removed {track_analysis['track_not_found']} from dataset")
            continue
        count += 1
        spinner.start(f'Fetching tracks ({count / _track_list.get_desired_tracks_amount():.2f}%)')
        extracted = extract_track_analysis(track_analysis)
        store_extracted_analysis(output_dir, extracted)
    spinner.stop()
    _track_list.dump(output_dir)


def get_missing(output_dir, data_type):
    _track_list = TrackList.load_from_dir(output_dir)
    return [
        track_id
        for track_id in _track_list.get_track_ids()
        if not have_datapoint(output_dir, data_type, track_id)
    ]


def get_obsolete(output_dir, data_type):
    _track_list = TrackList.load_from_dir(output_dir)
    return [
        str(track_file.name).split('.')[0]
        for track_file in Path(get_data_dir(output_dir, data_type)).glob('*.pickle')
        if not _track_list.have_track_id(str(track_file.name).split('.')[0])
    ]


def missing(output_dir, data_type, absolute):
    missing_ids = get_missing(output_dir, data_type)
    if absolute:
        missing_ids = [
            join(getcwd(), get_track_data_path(output_dir, data_type, missing_id))
            for missing_id in missing_ids
        ]
    for missing_id in missing_ids:
        print(missing_id)


def obsolete(output_dir, data_type, absolute):
    obsolete_ids = get_obsolete(output_dir, data_type)
    if absolute:
        obsolete_ids = [
            join(getcwd(), get_track_data_path(output_dir, data_type, obsolete_id))
            for obsolete_id in obsolete_ids
        ]
    for obsolete_id in obsolete_ids:
        print(obsolete_id)


def check(output_dir):
    for data_type in [AUDIO_ANALYSIS, AUDIO_FEATURES]:
        missing_ids = get_missing(output_dir, data_type)
        if len(missing_ids) == 0:
            print(f"Everything is fine for {data_type}")
        else:
            print(f'missing {data_type} for {len(missing_ids)} tracks')


def count(output_dir):
    track_list = TrackList.load_from_dir(output_dir)
    N = len(track_list.track_ids)
    print(f'expecting {N} tracks')
    N_feat = count_data_points(output_dir, AUDIO_FEATURES)
    print(f'found {N_feat} features objects')
    N_ana = count_data_points(output_dir, AUDIO_ANALYSIS)
    print(f'found {N_ana} analysis objects')


if __name__ == '__main__':
    args = get_args()
    if args.command == 'list':
        track_list = None
        if args.use_list:
            track_list = TrackList.load(args.use_list)
            args.N = track_list.get_desired_tracks_amount()
        else:
            output_dir = (Path(getcwd()) / args.output_dir).absolute()
            if len(list(Path(get_data_dir(args.output_dir, AUDIO_FEATURES)).glob('*.pickle'))) > 0:
                print(f'there already some audio features downloaded and stored in the output directory '
                      f'({output_dir}). Either provide the path to a track_list.pickle with --use-list or make '
                      f'sure that {output_dir} is empty')
                exit()
        list_tracks(args.mpl_dir, args.output_dir, args.N, args.list_dir, track_list)
    elif args.command == 'fetch':
        fetch(args.output_dir)
    elif args.command == 'check':
        check(args.output_dir)
    elif args.command == 'count':
        count(args.output_dir)
    elif args.command == 'obsolete':
        obsolete(args.output_dir, args.data_type, args.absolute)
    elif args.command == 'missing':
        missing(args.output_dir, args.data_type, args.absolute)
