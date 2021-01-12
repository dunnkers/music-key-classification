from argparse import ArgumentParser

from constants import AUDIO_FEATURES, AUDIO_ANALYSIS


def add_list_parser(sub_parsers):
    list_sub_parser = sub_parsers.add_parser('list', help='''
        Create a list of tracks, balanced on key and mode and put it in the meta object. At the same time, fetch the 
        audio_features objects belonging to the tracks
    ''')
    list_sub_parser.add_argument('--mpl-dir', default='spotify_million_playlist_dataset/data', type=str, help='''
        The location of the Spotify Million Playlist Dataset
    ''')
    list_sub_parser.add_argument('--use-list', default='', type=str, help='''
        The path to a pickled track list.
    ''')
    list_sub_parser.add_argument('--list-dir', default='', type=str, help='''
        The path to store the track list as a .pickle file.
    ''')
    list_sub_parser.add_argument('N', type=int, help='''
        The amount of tracks to fetch.
    ''')


def add_fetch_parser(sub_parsers):
    fetch_sub_parser = sub_parsers.add_parser('fetch', help='''
        Resume fetching data from the Spotify API to put into a dataset. Expects OUTPUT_DIR to contain a valid meta file
    ''')


def add_check_parser(sub_parsers):
    check_sub_parser = sub_parsers.add_parser('check', help='''
        Check if the data in OUTPUT_DIR is complete and valid
    ''')


def add_count_parser(sub_parsers):
    count_sub_parser = sub_parsers.add_parser('count', help='''
        Check if the data in OUTPUT_DIR is complete and valid
    ''')


def add_missing_parser(sub_parsers):
    check_sub_parser = sub_parsers.add_parser('missing', help='''
            List missing data points
        ''')
    check_sub_parser.add_argument(
        'data-type',
        default=AUDIO_ANALYSIS,
        choices=[AUDIO_ANALYSIS, AUDIO_FEATURES],
        type=str,
        help='''The type of data point to list missing for'''
    )


def add_obsolete_parser(sub_parsers):
    obsolete_sub_parser = sub_parsers.add_parser('obsolete', help='''
            List obsolete data points
        ''')
    obsolete_sub_parser.add_argument(
        'data-type',
        default=AUDIO_ANALYSIS,
        choices=[AUDIO_ANALYSIS, AUDIO_FEATURES],
        type=str,
        help='''The type of data point to list missing for'''
    )


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--output-dir', default='dataset', type=str, help='''
        The directory where the fetched data should be stored
        ''')
    sub_parsers = arg_parser.add_subparsers(dest='command')
    add_list_parser(sub_parsers)
    add_fetch_parser(sub_parsers)
    add_count_parser(sub_parsers)
    add_check_parser(sub_parsers)
    add_missing_parser(sub_parsers)
    add_obsolete_parser(sub_parsers)
    return arg_parser.parse_args()
