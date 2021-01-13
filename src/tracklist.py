from os.path import join
from pathlib import Path
from dill import dump, load


class TrackList:
    def __init__(self):
        self.track_ids = []
        self.track_id_set = set([])

    def get_track_ids(self):
        return self.track_ids

    def set_track_ids(self, track_ids: list):
        self.track_ids = track_ids
        self.track_id_set = set(track_ids)

    def add_track_id(self, track_id: str):
        self.track_ids.append(track_id)
        self.track_id_set.add(track_id)

    def have_track_id(self, track_id: str):
        return track_id in self.track_id_set

    def remove_track_id(self, track_id):
        self.track_ids.remove(track_id)
        self.track_id_set.remove(track_id)

    def dump(self, output_dir):
        with open(join(output_dir, 'track_list.pickle'), 'wb') as f:
            dump(self, f)

    def ids(self):
        for track_id in self.track_ids:
            yield track_id

    def hash(self):
        return hash(''.join(sorted(self.track_ids)))

    @classmethod
    def load_from_dir(cls, output_dir):
        return TrackList.load(Path(output_dir) / 'track_list.pickle')

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return load(f)
