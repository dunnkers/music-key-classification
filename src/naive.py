from argparse import ArgumentParser
from tracklist import TrackList
from data import load_analysis
import numpy as np

key_nums = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
scales = {
    #      C  C#  D  D#  E  F  F#  G  G#  A  A#  B
    'C':  [1, 0,  1, 0,  1, 1, 0,  1, 0,  1, 0,  1],
    'C#': [1, 1,  0, 1,  0, 1, 1,  0, 1,  0, 1,  0],
    'D':  [0, 1,  1, 0,  1, 0, 1,  1, 0,  1, 0,  1],
    'D#': [1, 0,  1, 1,  0, 1, 0,  1, 1,  0, 1,  0],
    'E':  [0, 1,  0, 1,  1, 0, 1,  0, 1,  1, 0,  1],
    'F':  [1, 0,  1, 0,  1, 1, 0,  1, 0,  1, 1,  0],
    'F#': [0, 1,  0, 1,  0, 1, 1,  0, 1,  0, 1,  1],
    'G':  [1, 0,  1, 0,  1, 0, 1,  1, 0,  1, 0,  1],
    'G#': [1, 1,  0, 1,  0, 1, 0,  1, 1,  0, 1,  0],
    'A':  [0, 1,  1, 0,  1, 0, 1,  0, 1,  1, 0,  1],
    'A#': [1, 0,  1, 1,  0, 1, 0,  1, 0,  1, 1,  0],
    'B':  [0, 1,  0, 1,  1, 0, 1,  0, 1,  0, 1,  1]
}


def compute_accuracy(conf_mat):
    return np.sum(np.diag(conf_mat))/np.sum(conf_mat)


def estimate_key(analysis):
    ''' Naive key estimation routine.
    The important data is first reformatted to a Numpy array, then processed.
    This method compares the weighted (in this case, by duration) average chroma key 
    for every segment with each scale's notes.
    
    Returns the index number of the estimated key.
    '''

    # Reformat data
    num_segments = len(analysis["segments"])
    input_data = np.zeros((num_segments, 13))
    i = 0
    for segment in analysis["segments"]:
        input_data[i,0]  = segment["duration"]
        input_data[i,1:] = segment["pitches"]
        i += 1
    
    # Compare weighted average chroma key with every scale
    avg_vec = np.average(input_data[:,1:], axis=0, weights=input_data[:,0])
    scores = np.zeros(12)
    for i in range(0, 12):
        scores[i] = np.correlate(avg_vec, scales[key_nums[i]])
    return np.argmax(scores)


def run_naive_method_all_tracks(data_dir):
    ''' Runs the naive method on all stored track analysis data.
    Uses the `estimate_key` function to estimate the key of the track
    and computes a confusion matrix.
    '''

    meta = TrackList.load_from_dir(data_dir)
    all_tracks = meta.get_track_ids()

    i = 0
    conf_mat = np.zeros((12,12))
    n_10perc   = int(len(all_tracks)/10)
    for track_id in meta.get_track_ids():
        analysis        = load_analysis(data_dir, track_id)

        if analysis["mode"] == 1:
            correct_answer  = analysis["key"]
            estimate        = estimate_key(analysis)
            conf_mat[estimate, correct_answer] += 1
        
        i += 1
        if i % n_10perc == 0:
            print(f"{int(i/n_10perc)}0%")
    return conf_mat
        


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-dir', default='dataset', type=str, help='''
        The directory where the track data is stored to use for the analysis
        ''')
    arg_parser.add_argument('--csv', default=False, type=str, help='''
        Optional filename of a CSV file to store the resulting confusion matrix
        ''')
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    conf_mat = run_naive_method_all_tracks(args.data_dir)
    print(conf_mat)
    print(f"N=%d" % (np.sum(conf_mat)))
    print(f"Overall accuracy: %.2f%%" % (100*compute_accuracy(conf_mat)))
    if args.csv is not False:
        np.savetxt(args.csv, conf_mat, delimiter=",", fmt="%d")
