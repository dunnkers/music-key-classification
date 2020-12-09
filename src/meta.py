from os.path import join
from pickle import dump, load


class Meta:
    def get_track_ids(self):
        return self.track_ids

    def set_track_ids(self, track_ids):
        self.track_ids = track_ids

    def add_track_id(self, track_id):
        self.track_ids.append(track_id)

    def dump(self, output_dir):
        with open(join(output_dir, 'meta.pickle'), 'wb') as f:
            dump(self.__dict__, f)

    @classmethod
    def load(cls, output_dir):
        meta = Meta()
        with open(join(output_dir, 'meta.pickle'), 'rb') as f:
            data = load(f)
        meta.set_track_ids(data['track_ids'])
        return meta
