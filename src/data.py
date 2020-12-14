from argparse import ArgumentParser
from os import makedirs
from os.path import exists, isdir, join
from shutil import rmtree
from pickle import dump, load

from meta import Meta
from mpl import list_track_ids
from track_analysis import n_track_analyses_generator
from process import extract_track_analysis


def get_meta_path(output_dir):
    return join(output_dir, 'meta.pickle')


def get_track_path(output_dir, track_id):
    return join(output_dir, track_id) + '.pickle'


def store_extracted_analysis(output_dir, extracted_track_analysis):
    with open(get_track_path(output_dir, extracted_track_analysis['id']), 'wb+') as f:
        dump(extracted_track_analysis, f)

def load_analysis(output_dir, track_id):
    with open(get_track_path(output_dir, track_id), 'rb') as f:
        return load(f)


def create_meta(track_ids):
    meta = Meta()
    meta.set_track_ids(track_ids)
    return meta


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
        extracted = extract_track_analysis(track_analysis)
        store_extracted_analysis(output_dir, extracted)
    print('done fetching')


def resume_fetching(output_dir):
    print("resuming fetching...")
    meta = Meta.load(output_dir)
    track_ids = [track_id for track_id in meta.get_track_ids() if not exists(get_track_path(output_dir, track_id))]
    track_analyses = n_track_analyses_generator(track_ids)
    for track_analysis in track_analyses:
        extracted = extract_track_analysis(track_analysis)
        store_extracted_analysis(output_dir, extracted)
    print('done fetching')


def get_missing(output_dir):
    meta = Meta.load(output_dir)
    return [track_id for track_id in meta.get_track_ids() if not exists(get_track_path(output_dir, track_id))]


def check(output_dir, output_missing):
    missing = get_missing(output_dir)
    if len(missing) == 0:
        print("Everything is fine")
    elif output_missing == 'list':
        for track_id in missing:
            print(track_id)
    else:
        print(f'missing {len(missing)} tracks')


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--output-dir', default='dataset', type=str, help='''
        The directory where the fetched data should be stored
        ''')
    sub_parsers = arg_parser.add_subparsers(dest='command')
    # Fetch command
    fetch_sub_parser = sub_parsers.add_parser('fetch', help='''
    Fetch data from the Spotify API to put into a dataset
    ''')
    fetch_sub_parser.add_argument('--output-dir', default='dataset', type=str, help='''
    The directory where the fetched data should be stored
    ''')
    fetch_sub_parser.add_argument('--mpl-dir', default='spotify_million_playlist_dataset/data', type=str, help='''
    The location of the Spotify Million Playlist Dataset
    ''')
    fetch_sub_parser.add_argument('N', type=int, help='''
    The amount of tracks to fetch
    ''')
    # Resume command
    resume_sub_parser = sub_parsers.add_parser('resume', help='''
    Resume fetching data from the Spotify API to put into a dataset. Expects OUTPUT_DIR to contain a valid meta file
    ''')
    # Check command
    check_sub_parser = sub_parsers.add_parser('check', help='''
    Check if the data in OUTPUT_DIR is complete and valid
    ''')
    check_sub_parser.add_argument('--output-type', type=str, default='count', choices=['count', 'list'], help='''
    Output the count or the list of missing track analyses
    ''')
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.command == 'fetch':
        start_fetching(args.mpl_dir, args.output_dir, args.N)
    elif args.command == 'resume':
        resume_fetching(args.output_dir)
    elif args.command == 'check':
        check(args.output_dir, args.output_type)
