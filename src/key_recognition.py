from argparse import ArgumentParser
from meta import Meta
from data import load_analysis
import numpy as np
from tabulate import tabulate

key_nums = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
modes = ["Minor", "Major"]

def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-dir', default='dataset', type=str, help='''
        The directory where the track data is stored to use for the analysis
        ''')
    arg_parser.add_argument('--csv', default=False, type=str, help='''
        Optional filename of a CSV file to store the resulting confusion matrix [NOT YET IMPLEMENTED]
        ''')
    arg_parser.add_argument('--test-split', default=0.2, type=float, help='''
        The fraction of samples to use as testing data
        ''')
    arg_parser.add_argument('--table', action='store_true', help='''
        Whether or not to print a table of all the test samples and their classification
        ''')
    sub_parsers = arg_parser.add_subparsers(dest='method')
    sub_parsers.required = True

    # Naive method
    naive_sub_parser = sub_parsers.add_parser('naive', help='''
        Test classification using the naive method.
    ''')
    naive_sub_parser.add_argument('--no-training', action='store_true', help='''
        Don't train the naive model, just use the pre-programmed correlation vectors
        ''')

    # HMM method
    hmm_sub_parser = sub_parsers.add_parser('hmm', help='''
        Test classification using the HMM method.
    ''')
    return arg_parser.parse_args()


def load_data_dict(data_dir, track_ids, mode=False):
    testing_data = {}
    for track_id in track_ids:
        analysis = load_analysis(data_dir, track_id)
        if not mode:
            testing_data[track_id] = analysis
        elif analysis["mode"] == mode:
            testing_data[track_id] = analysis
    return testing_data

def collect_data(data_dir, test_split, mode=False):
    meta = Meta.load(data_dir)
    all_tracks = meta.get_track_ids()
    n = len(all_tracks)
    train_n = int(n*(1-test_split))
    print("Collecting training data...")
    training_data = load_data_dict(data_dir, all_tracks[:train_n], mode=mode)
    print("Collecting testing data...")
    testing_data = load_data_dict(data_dir, all_tracks[train_n:], mode=mode)
    print("Data collected.")
    return training_data, testing_data


''' MAIN PROGRAM '''

def run_key_recognition(args):
    if args.method == 'naive':
        from naive_model import Naive_model
        model = Naive_model()
        pass
    else:
        from hmm_model import HMM_model
        model = HMM_model()
        pass
    
    # Collect data
    training_data, testing_data = collect_data(args.data_dir, args.test_split, mode=1)

    # Train model
    if not args.no_training:
        model.train(training_data)

    # Test model
    results_table = []
    test_n = 0
    errors = 0
    for track_id in testing_data:
        track_data = testing_data[track_id]
        estimation_key = model.predict(track_data)

        # Count errors
        test_n += 1
        if not (track_data["key"] == estimation_key % 12 and track_data["mode"] == estimation_key // 12):
            errors += 1
        
        # Report all test samples
        results_table.append([track_id, 
            "%s %s"% (key_nums[track_data["key"]], modes[track_data["mode"]]), 
            "%s %s"% (key_nums[estimation_key % 12], modes[estimation_key // 12]),
        ])
    return errors/test_n, results_table

if __name__ == '__main__':
    args = get_args()
    error, results_table = run_key_recognition(args)
    if args.table:
        print(tabulate(results_table, headers=["Song ID", "Label key", "Predicted key"]))
    print("Overall error: %5.2f%%" % (error*100))
