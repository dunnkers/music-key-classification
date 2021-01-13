import operator
from functools import reduce
from math import ceil
from os import makedirs
from os.path import exists, join
from pathlib import Path
from dill import dump, load

from args import get_args
from constants import AUDIO_ANALYSIS, AUDIO_FEATURES
from tracklist import TrackList
from mpl import track_id_generator
from track_analysis import n_track_analyses_generator
from process import extract_track_analysis, extract_audio_features
from track_features import n_track_features


def get_meta_path(output_dir) -> str:
    return join(output_dir, 'meta.pickle')


def get_data_dir(output_dir, data_type) -> str:
    return join(output_dir, data_type)


def get_track_data_path(output_dir, data_type, track_id) -> str:
    return join(get_data_dir(output_dir, data_type), track_id) + '.pickle'


def get_audio_analysis_path(output_dir, track_id) -> str:
    return get_track_data_path(output_dir, AUDIO_ANALYSIS, track_id)


def get_audio_features_path(output_dir, track_id) -> str:
    return get_track_data_path(output_dir, AUDIO_FEATURES, track_id)


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


def load_features(output_dir, track_id) -> dict:
    return load_datapoint(output_dir, AUDIO_FEATURES, track_id)


def load_analysis(output_dir, track_id) -> dict:
    return load_datapoint(output_dir, AUDIO_ANALYSIS, track_id)


def create_meta(track_ids) -> TrackList:
    meta = TrackList()
    meta.set_track_ids(track_ids)
    return meta


def get_n_track_ids(track_id_gen, N):
    track_ids = []
    next_id = next(track_id_gen, None)
    while next_id is not None and len(track_ids) < N:
        track_ids.append(next_id)
        next_id = next(track_id_gen, None)
    return track_ids


def list_tracks(mpl_data_path, output_dir, N, list_dir = '', track_list: TrackList = None) -> None:
    if not exists(get_data_dir(output_dir, AUDIO_FEATURES)):
        makedirs(get_data_dir(output_dir, AUDIO_FEATURES))
    # We will count the amount of tracks per key and mode
    key_counts = dict()
    required_per_key = ceil(N / 24)
    for key in range(24):
        key_counts[key] = 0
    # Start from a provided track list or create a new one
    track_list_complete = False
    if track_list is not None:
        track_list_complete = True
        track_id_gen = track_list.ids()
    else:
        track_id_gen = track_id_generator(mpl_data_path)
        track_list = TrackList()
    finished = False
    total = 0
    while not finished:
        track_ids = get_n_track_ids(track_id_gen, 100)
        track_feats = n_track_features(track_ids)
        for track_feat in track_feats:
            extracted_track_features = extract_audio_features(track_feat)
            key = extracted_track_features['key'] + (extracted_track_features['mode'] * 12)
            if key_counts[key] < required_per_key:
                key_counts[key] += 1
                if not track_list_complete:
                    track_list.add_track_id(extracted_track_features['id'])
                store_extracted_features(output_dir, extracted_track_features)
                total += 1
            finished = reduce(operator.and_, [key_counts[key] >= required_per_key for key in range(24)]) or total >= N
            if finished:
                break
    track_list.dump(output_dir)
    if list_dir:
        track_list.dump(list_dir)


def fetch(output_dir):
    meta = TrackList.load_from_dir(output_dir)
    track_ids = [track_id for track_id in meta.get_track_ids() if not exists(get_audio_analysis_path(output_dir, track_id))]
    if not exists(get_data_dir(output_dir, AUDIO_ANALYSIS)):
        makedirs(get_data_dir(output_dir, AUDIO_ANALYSIS))
    track_analyses = n_track_analyses_generator(track_ids)
    for track_analysis in track_analyses:
        if 'track_not_found' in track_analysis:
            meta.remove_track_id(track_analysis['track_not_found'])
            print(f"removed {track_analysis['track_not_found']} from dataset")
            continue
        extracted = extract_track_analysis(track_analysis)
        store_extracted_analysis(output_dir, extracted)
    meta.dump(output_dir)


def get_missing(output_dir, data_type):
    meta = TrackList.load_from_dir(output_dir)
    return [track_id for track_id in meta.get_track_ids() if not exists(get_track_data_path(output_dir, data_type, track_id))]


def get_obsolete(output_dir, data_type):
    meta = TrackList.load_from_dir(output_dir)
    return [
        str(track_file.name).split('.')[0]
        for track_file in Path(get_data_dir(output_dir, data_type)).glob('*.pickle')
        if not meta.have_track_id(str(track_file.name).split('.')[0])
    ]


def missing(output_dir, data_type):
    missing_ids = get_missing(output_dir, data_type)
    for missing_id in missing_ids:
        print(missing_id)


def obsolete(output_dir, data_type):
    obsolete_ids = get_obsolete(output_dir, data_type)
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
    N_feat = len(list(Path(get_data_dir(output_dir, AUDIO_FEATURES)).glob('*.pickle')))
    print(f'found {N_feat} features objects')
    N_ana = len(list(Path(get_data_dir(output_dir, AUDIO_ANALYSIS)).glob('*.pickle')))
    print(f'found {N_ana} analysis objects')


if __name__ == '__main__':
    args = get_args()
    if args.command == 'list':
        track_list = None
        if args.use_list:
            track_list = TrackList.load(args.use_list)
            args.N = len(track_list.track_ids)
        list_tracks(args.mpl_dir, args.output_dir, args.N, args.list_dir, track_list)
    elif args.command == 'fetch':
        fetch(args.output_dir)
    elif args.command == 'check':
        check(args.output_dir)
    elif args.command == 'count':
        count(args.output_dir)
    elif args.command == 'obsolete':
        obsolete(args.output_dir, args.data_type)
    elif args.command == 'missing':
        missing(args.output_dir, args.data_type)
