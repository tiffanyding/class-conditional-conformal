import argparse

from utils.experiment_utils import *


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('dataset', type=str, choices=['imagenet', 'cifar-100', 'places365', 'inaturalist'],
                        help='Name of the dataset to train model on')
    parser.add_argument('avg_num_per_class', type=int,
                        help='Number of examples per class, on average, to include in calibration dataset')
    parser.add_argument('-score_functions', type=str,  nargs='+', 
                        help='Conformal score functions to use. List with a space in between. Options are'
                        '"softmax", "APS", "RAPS"')
    parser.add_argument('-methods', type=str,  nargs='+', 
                        help='Conformal methods to use. List with a space in between. Options include'
                        '"standard", "classwise", "classwise_default_standard", "always_cluster"')
    parser.add_argument('-seeds', type=int,  nargs='+', 
                        help='Seeds for random splits into calibration and validation sets,'
                        'List with spaces in between')
    

    parser.add_argument('--calibration_sampling', type=str, default='random',
                    help='How to sample the calibration set. Options are "random" and "balanced"')
    parser.add_argument('--frac_clustering', type=float, default=-1,
                    help='For clustered conformal: the fraction of data used for clustering.'
                       'If frac_clustering and num_clusters are both -1, then a heuristic will be used to choose these values.')
    parser.add_argument('--num_clusters', type=int, default=-1,
                    help='For clustered conformal: the number of clusters to request'
                       'If frac_clustering and num_clusters are both -1, then a heuristic will be used to choose these values.')
    parser.add_argument('--alpha', type=float, default=0.1,
                    help='Desired coverage is 1-alpha')
    parser.add_argument('--save_folder', type=str, default='.cache/paper/varying_n',
                        help='Folder to save results to')

    
    args = parser.parse_args()
    if args.frac_clustering != -1 and args.num_clusters != -1:
        run_one_experiment(args.dataset, args.save_folder, args.alpha, 
                           args.avg_num_per_class, args.score_functions, args.methods, args.seeds, 
                           cluster_args={'frac_clustering': args.frac_clustering, 'num_clusters': args.num_clusters}, 
                           save_preds=False, calibration_sampling=args.calibration_sampling)
    else: # choose frac_clustering and num_clusters automatically 
        run_one_experiment(args.dataset, args.save_folder, args.alpha, 
                           args.avg_num_per_class, args.score_functions, args.methods, args.seeds, 
                           save_preds=False, calibration_sampling=args.calibration_sampling)