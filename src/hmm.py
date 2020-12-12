from pickle import load
from meta import Meta
from data import get_meta_path, get_track_path

output_dir = './dataset'
meta = Meta.load(output_dir)
print(get_meta_path(output_dir))

def load_extracted_analysis(output_dir, track_id):
    path = get_track_path(output_dir, track_id)
    with open(path, 'rb') as f:
        return load(f)

def load_track(track_id):
    return load_extracted_analysis(output_dir, track_id)

X = list(map(load_track, meta.track_ids))
print('..')