from argparse import ArgumentParser
from tracklist import TrackList
from data import load_analysis
import numpy as np
from tabulate import tabulate

key_nums = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
modes = ["Minor", "Major"]

def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-dir', default='dataset', type=str, help='''
        The directory where the track data is stored to use for the analysis.
        ''')
    arg_parser.add_argument('--give-mode', action='store_true', help='''
        Optionally test the model with given mode (major/minor).
        ''')
    arg_parser.add_argument('--test-split', default=5, type=int, help='''
        Use 1/N of the data for test validation
        ''')
    arg_parser.add_argument('--csv', default=False, type=str, help='''
        Optional filename of a CSV file to store the resulting confusion matrix
        ''')
    arg_parser.add_argument('--table', action='store_true', help='''
        Whether or not to print a table of all the test samples and their classification
        ''')
    arg_parser.add_argument('--verbose', action='store_true', help='''
        Verbose model training
        ''')
    arg_parser.add_argument('--cross-validation', action='store_true', help='''
        Run N-fold cross-validation (N = <--test-split>)
        ''')
    arg_parser.add_argument('--dry', action='store_true', help='''
        Dry run: only test program execution - use only a couple training samples in total.
        ''')
    arg_parser.add_argument('--subset', default=10000, type=int, help='''
        Subset sample: only use a subset of training samples in total.
        ''')
    arg_parser.add_argument('--n_components', default=3, type=int, help='''
        Amount of components ('hidden states') to use for HMM training.
        ''')
    arg_parser.add_argument('--n_iter', default=100, type=int, help='''
        (Maximum) amount of iterations used in HMM training.
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
    hmm_sub_parser.add_argument('--mixture', action='store_true', help='''
        Use a Gaussian mixture model
        ''')
    return arg_parser.parse_args()


def load_data_dict(data_dir, track_ids):
    testing_data = {}
    for track_id in track_ids:
        analysis = load_analysis(data_dir, track_id)
        testing_data[track_id] = analysis
    return testing_data

def collect_data(data_dir, test_split, test_split_index=0 , verbose=False, dry=False, subset=10000):
    track_list = TrackList.load_from_dir(data_dir)
    all_tracks = track_list.get_track_ids()
    n = len(all_tracks)
    if dry: # use virtually no data at all - just test the program execution
        n = subset
    chunks = np.array_split(np.arange(n), test_split)
    test_split =  chunks[test_split_index]
    train_split = np.concatenate(chunks[:test_split_index] + chunks[test_split_index+1:])
    if verbose:
        print("Collecting training data...")
    training_data = load_data_dict(data_dir, np.array(all_tracks)[train_split])
    if verbose:
        print("Collecting testing data...")
    testing_data = load_data_dict(data_dir, np.array(all_tracks)[test_split])
    if verbose:
        print("Data collected.")
    return training_data, testing_data



''' MAIN PROGRAM '''
def run_key_recognition(args, verbose=True, test_split_index=0):

    # Import selected model
    if args.method == 'naive':
        from naive_model import Naive_model
        model = Naive_model()
        pass
    else:
        from hmm_model import HMM_model
        model = HMM_model(n_components=args.n_components, n_iter=args.n_iter)
        if args.mixture:
            model.mixture = True
        pass
    
    # Collect data
    training_data, testing_data = collect_data(args.data_dir, args.test_split, test_split_index=test_split_index, verbose=verbose, dry=args.dry, subset=args.subset)

    # Train model
    if args.method != 'naive' or not args.no_training:
        model.train(training_data, verbose=verbose)
    
    results_table = []
    test_n = 0
    errors = 0
    conf_mat = np.zeros((24,24))

    # Try all the testing samples on the model
    if verbose:
        print("Testing model...")
    for track_id in testing_data:
        track_data = testing_data[track_id]
        if args.give_mode:
            estimation_key = model.predict(track_data, mode=track_data["mode"])
        else:
            estimation_key = model.predict(track_data)
        
        # Confusion matrix
        conf_mat[estimation_key, track_data["mode"]*12 + track_data["key"]] += 1

        # Count errors
        test_n += 1
        if not (track_data["key"] == estimation_key % 12 and track_data["mode"] == estimation_key // 12):
            errors += 1
        
        # Report all test samples
        results_table.append([track_id, 
            "%s %s"% (key_nums[track_data["key"]], modes[track_data["mode"]]), 
            "%s %s"% (key_nums[estimation_key % 12], modes[estimation_key // 12]),
        ])
    if verbose:
        print("Done.")

    return errors/test_n, results_table, conf_mat

if __name__ == '__main__':
    args = get_args()
    if args.cross_validation:
        print(f"Running {args.test_split}-fold cross validation.")
        for i in range(args.test_split):
            error, results_table, confusion_matrix = run_key_recognition(args, verbose=False, test_split_index=i)
        
            if args.table:
                print(tabulate(results_table, headers=["Song ID", "Label key", "Predicted key"]))
                print(confusion_matrix)
            
            print("Overall error: %5.2f%%" % (error*100))
        
            if args.csv is not False:
                np.savetxt('split_{}-{}'.format(i, args.csv), confusion_matrix)

    else:
        error, results_table, confusion_matrix = run_key_recognition(args, verbose=args.verbose)
        
        if args.table:
            print(tabulate(results_table, headers=["Song ID", "Label key", "Predicted key"]))
            print(confusion_matrix)
        
        print("Overall error: %5.2f%%" % (error*100))
        
        if args.csv is not False:
            np.savetxt(args.csv, confusion_matrix)
