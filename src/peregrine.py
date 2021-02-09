from key_recognition import run_key_recognition
from argparse import Namespace, ArgumentParser
from tabulate import tabulate
import numpy as np
import multiprocessing 

if __name__ == '__main__':
    # Get hyperargs
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--n_components', default=3, type=int, help='''
        Amount of components ('hidden states') to use for HMM training.
        ''')
    arg_parser.add_argument('--n_iter', default=100, type=int, help='''
        (Maximum) amount of iterations used in HMM training.
        ''')
    arg_parser.add_argument('--subset', default=1000, type=int, help='''
        Samples to use.
        ''')
    hyperargs = arg_parser.parse_args()

    # Get args - manually create Namespace object
    args = Namespace()
    args.data_dir = 'dataset'
    args.dry = False
    args.table = True
    args.verbose = True
    args.give_mode = True
    args.method = 'hmm'
    args.mixture = False
    args.cross_validation = True
    args.test_split = 10
    args.n_components = hyperargs.n_components
    args.n_iter = hyperargs.n_iter
    args.subset = hyperargs.subset
    args.csv = True

    # Function to run on 1 fold
    def run_fold(fold):
        print(f"[Split {fold}]")
        error, results_table, confusion_matrix = run_key_recognition(\
            args, verbose=False, test_split_index=fold)
        print("Overall error: %5.2f%%" % (error*100))
        if args.table:
            print(tabulate(results_table,\
                headers=["Song ID", "Label key","Predicted key"]))
            print(confusion_matrix)
        if args.csv is not False:
            filename = 'logs/n_components={},n_iter={},fold={}.csv'\
                .format(args.n_components, args.n_iter, fold)
            np.savetxt(filename, confusion_matrix)

    # Run k-fold CV in parallel
    print(f"Running {args.test_split}-fold CV in parallel.")
    print(f"[n_components={args.n_components}, n_iter={args.n_iter}]")
    cpus = multiprocessing.cpu_count()
    processes = cpus if cpus < 10 else 10
    pool = multiprocessing.Pool(processes=processes,)
    pool.map(run_fold, range(args.test_split))
    pool.close()
    pool.join()
    print('Done!')