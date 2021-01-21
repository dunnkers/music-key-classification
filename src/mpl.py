from os.path import join
from os import listdir
from json import load


def track_id_generator(mpl_data_path, have_id=None):
    files = listdir(mpl_data_path)
    for file in files:
        # with open(join(mpl_data_path, file), 'r') as f:
        with open(join(mpl_data_path, file), 'r', encoding='utf-8') as f:
            print(f)
            mpl_slice = load(f)
            for playlist in mpl_slice['playlists']:
                for track in playlist['tracks']:
                    yield track['track_uri'].split(':')[2]


def list_track_ids(mpl_data_path, n_tracks=100):
    """
    1.3 seconds to list 100k
    > 60 seconds to list 1M
    """
    track_ids_gen = track_id_generator(mpl_data_path)
    track_ids = set()
    while len(track_ids) < n_tracks:
        track_ids.add(next(track_ids_gen))
    return list(track_ids)
