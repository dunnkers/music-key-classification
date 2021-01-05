from argparse import ArgumentParser
from meta import Meta
from data import load_analysis
import numpy as np
from hmmlearn import hmm
from tabulate import tabulate

key_nums = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def format_sequence(audio_analysis):
    num_segments = len(audio_analysis["segments"])
    seq = np.zeros((num_segments, 12))
    i = 0
    for segment in audio_analysis["segments"]:
        seq[i,:] = segment["pitches"]
        i += 1
    return seq

def run_hmm_method_all_tracks(data_dir, test_split=0.2):
    ''' Trains the HMM 
    '''

    meta = Meta.load(data_dir)
    all_tracks = meta.get_track_ids()
    n = len(all_tracks)
    train_n = int(n*(1-test_split))

    print("Collecting training data...")
    sequences        = np.zeros((0,12))
    sequence_lengths = []
    for track_id in all_tracks[:train_n]:
        analysis = load_analysis(data_dir, track_id)
        
        # Format sequence
        seq = format_sequence(analysis)
        seq = np.roll(seq, -analysis["key"])
        
        # Add sequence to set of all sequences
        sequences = np.concatenate((sequences, seq))
        sequence_lengths.append(seq.shape[0])
    sequence_lengths = np.array(sequence_lengths)
    
    print("Training model...")
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    model.fit(sequences, sequence_lengths)
    print("Done.")
    
    print("Testing model...")
    print("")

    results = []
    for track_id in all_tracks[train_n:]:
        analysis = load_analysis(data_dir, track_id)
        seq = format_sequence(analysis)
        prob = model.score(seq)
        results.append([track_id, key_nums[analysis["key"]], prob])
    print(tabulate(results, headers=["Song ID", "Key label", "HMM probability of C"]))

        


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-dir', default='dataset', type=str, help='''
        The directory where the track data is stored to use for the analysis
        ''')
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_hmm_method_all_tracks(args.data_dir)
    