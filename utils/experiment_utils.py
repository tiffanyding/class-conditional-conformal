import glob # For getting file names
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

import pdb

from collections import Counter
from scipy import stats, cluster

from utils.clustering_utils import *
from utils.conformal_utils import *

# For plotting
map_to_label = {'standard': 'Standard', 
               'classwise': 'Classwise',
               'always_cluster': 'Clustered'}
map_to_color = {'standard': 'gray', 
               'classwise': 'lightcoral',
               'always_cluster': 'dodgerblue'}


def remove_rare_classes(softmax_scores, labels, thresh = 150):
    '''
    Filter out classes with fewer than 150 examples
    
    Note: Make sure to raw softmax scores instead of 1-softmax in order for
    normalization to work correctly
    '''
    classes, cts = np.unique(labels, return_counts=True)
    non_rare_classes = classes[cts >= thresh]
    print(f'Data preprocessing: Keeping {len(non_rare_classes)} of {len(classes)} classes that have >= {thresh} examples')

    # Filter labels and re-index
    remaining_label_idx = np.isin(labels, non_rare_classes)
    labels = labels[remaining_label_idx]
    new_idx = 0
    mapping = {} # old to new
    for i, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = new_idx
            new_idx += 1
        labels[i] = mapping[label]
    
    # Remove rows and columns corresponding to rare classes from scores matrix
    softmax_scores = softmax_scores[remaining_label_idx,:]
    new_softmax_scores = np.zeros((len(labels), len(non_rare_classes)))
    for k in non_rare_classes:
        new_softmax_scores[:, mapping[k]] = softmax_scores[:,k]
    
    # Renormalize each row to sum to 1 
    new_softmax_scores = new_softmax_scores / np.expand_dims(np.sum(new_softmax_scores, axis=1), axis=1)

    return new_softmax_scores, labels



def load_dataset(dataset, remove_rare_cls=False):
    '''
    Inputs:
        - dataset: string specifying dataset. Options are ['imagenet', 'cifar-100', 'places365', 'inaturalist']
    '''
    if dataset == 'imagenet':
        softmax_path = '/home/tding/data/finetuned_imagenet/imagenet_train_subset_softmax.npy'
        labels_path = '/home/tding/data/finetuned_imagenet/imagenet_train_subset_labels.npy'
    elif dataset == 'cifar-100':
        softmax_path = "../class-conditional-conformal-datasets/notebooks/.cache/best-cifar100-model-valsoftmax_frac=0.3.npy"
        labels_path = "../class-conditional-conformal-datasets/notebooks/.cache/best-cifar100-model-vallabels_frac=0.3.npy"
    elif dataset == 'places365':
        softmax_path = '../class-conditional-conformal-datasets/notebooks/.cache/best-Places365-model-valsoftmax_frac=0.1.npy'
        labels_path = '../class-conditional-conformal-datasets/notebooks/.cache/best-Places365-model-vallabels_frac=0.1.npy'
    elif dataset == 'inaturalist':
        softmax_path = '../class-conditional-conformal-datasets/notebooks/.cache/best-iNaturalist-model-valsoftmax_frac=0.5.npy'
        labels_path = '../class-conditional-conformal-datasets/notebooks/.cache/best-iNaturalist-model-vallabels_frac=0.5.npy'
    
        remove_rare_cls = True

    softmax_scores = np.load(softmax_path)
    labels = np.load(labels_path)
    
    print('softmax_scores shape:', softmax_scores.shape) 
    
    if remove_rare_cls:
        softmax_scores, labels = remove_rare_classes(softmax_scores, labels, thresh=150)
    
    return softmax_scores, labels


def run_one_experiment(dataset, save_folder, alpha, n_totalcal, score_function_list, methods, seeds, 
                       save_preds=False, calibration_sampling='random'):
    '''
    Inputs:
        - dataset: string specifying dataset. Options are 'imagenet', 'cifar-100', 'places365', 'inaturalist'
        - n_totalcal: *average* number of examples per class. Calibration dataset is generated by sampling
          n_totalcal x num_classes examples uniformly at random
        - methods: List of conformal calibration methods. Options are 'standard', 'classwise', 
         'classwise_default_standard', 'cluster_balanced', 'cluster_proportional', 'cluster_doubledip'
        - save_preds: if True, the val prediction sets are included in the saved outputs
        - calibration_sampling: Method for sampling calibration dataset. Options are 
        'random' or 'balanced'
    '''
    
    softmax_scores, labels = load_dataset(dataset)
    
    for score_function in score_function_list:
        curr_folder = os.path.join(save_folder, f'{dataset}/{calibration_sampling}_calset/n_totalcal={n_totalcal}/score={score_function}')
        os.makedirs(curr_folder, exist_ok=True)

        print(f'====== score_function={score_function} ======')

        print('Computing conformal score...')
        if score_function == 'softmax':
            scores_all = 1 - softmax_scores
        elif score_function == 'APS':
            scores_all = get_APS_scores_all(softmax_scores, randomize=True)
        elif score_function == 'RAPS': 
            # RAPS hyperparameters (currently using ImageNet defaults)
            lmbda = .01 
            kreg = 5

            scores_all = get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True)
        else:
            raise Exception('Undefined score function')

        for seed in seeds:
            print(f'\nseed={seed}')
            save_to = os.path.join(curr_folder, f'seed={seed}_allresults.pkl')
            if os.path.exists(save_to):
                with open(save_to,'rb') as f:
                    all_results = pickle.load(f)
                    print('Loaded existing results file containing results for', list(all_results.keys()))
            else:
                all_results = {} # Each value is (qhat(s), preds, coverage_metrics, set_size_metrics)

            # Split data
            if calibration_sampling == 'random':
                totalcal_scores_all, totalcal_labels, val_scores_all, val_labels = random_split(scores_all, 
                                                                                                labels, 
                                                                                                n_totalcal, 
                                                                                                seed=seed)
            elif calibration_sampling == 'balanced':
                num_classes = scores_all.shape[1]
                totalcal_scores_all, totalcal_labels, val_scores_all, val_labels = split_X_and_y(scores_all, 
                                                                                                labels, n_totalcal, num_classes, seed=0, split='balanced')
            else:
                raise Exception('Invalid calibration_sampling option')
          
            # Inspect class imbalance of total calibration set
            cts = Counter(totalcal_labels).values()
            print(f'Class counts range from {min(cts)} to {max(cts)}')

            for method in methods:
                print(f'dataset={dataset}, n={n_totalcal},score_function={score_function}, seed={seed}, method={method}')

                if method == 'standard':
                    # Standard conformal
                    all_results[method] = standard_conformal_pipeline(totalcal_scores_all, totalcal_labels, 
                                                                          val_scores_all, val_labels, alpha)

                elif method == 'classwise':
                    # Classwise conformal  
                    all_results[method] = classwise_conformal_pipeline(totalcal_scores_all, totalcal_labels, 
                                                                       val_scores_all, val_labels, alpha, 
                                                                       default_qhat=np.inf, regularize=False)

                elif method == 'classwise_default_standard':
                    # Classwise conformal, but use standard qhat as default value instead of infinity 
                    all_results[method] = classwise_conformal_pipeline(totalcal_scores_all, totalcal_labels, 
                                                                       val_scores_all, val_labels, alpha, 
                                                                       default_qhat='standard', regularize=False)
                    
                elif method == 'cluster_balanced':
                    # Clustered conformal with balanced clustering set (does NOT provide cluster-conditional coverage)
                    all_results[method] = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                                    alpha,
                                                                            val_scores_all, val_labels, 
                                                                            split='balanced')
                elif method == 'cluster_proportional':
                    # Clustered conformal with proportionally sampled clustering set
                    all_results[method] = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                                    alpha,
                                                                            val_scores_all, val_labels, 
                                                                            split='proportional')
                
                elif method == 'cluster_doubledip':
                    # Clustered conformal with double dipping for clustering and calibration
                    all_results[method] = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                                    alpha,
                                                                            val_scores_all, val_labels, 
                                                                            split='doubledip')

                elif method == 'regularized_classwise':
                    
                    # Empirical-Bayes-inspired regularized classwise conformal (shrink class qhats to standard)
                    all_results[method] = classwise_conformal_pipeline(totalcal_scores_all, totalcal_labels, 
                                                                       val_scores_all, val_labels, alpha, 
                                                                       default_qhat='standard', regularize=True)
                else: 
                    raise Exception('Invalid method selected')

            # Optionally remove predictions from saved output to reduce memory usage
            if not save_preds:
                for m in all_results.keys():
                    all_results[m] = (all_results[m][0], None, all_results[m][2], all_results[m][3])

            # Save results 
            with open(save_to,'wb') as f:
                pickle.dump(all_results, f)
                print(f'Saved results to {save_to}')

# def run_experiment(softmax_scores, labels,
#                   save_folder,
#                   alpha=.1,
#                   n_totalcal_list=[10, 30],
#                   score_function_list = ['softmax', 'APS'],
#                   methods = ['standard', 'classwise', 'always_cluster'] # , 'regularized_classwise']
#                   seeds = [0,1,2,3,4],
#                   save_preds=False):
#     '''
#     If save_preds is True, the val prediction sets are included in the saved outputs
#     '''
    
#     num_classes = softmax_scores.shape[1]
    
#     for n_totalcal in n_totalcal_list:
#         for score_function in score_function_list:
#             curr_folder = os.path.join(save_folder, f'n_totalcal={n_totalcal}/score={score_function}')
#             os.makedirs(curr_folder, exist_ok=True)
            
#             print(f'====== score_function={score_function} ======')
    
#             print('Computing conformal score...')
#             if score_function == 'softmax':
#                 scores_all = 1 - softmax_scores
#             elif score_function == 'APS':
#                 scores_all = get_APS_scores_all(softmax_scores, randomize=True)
#             elif score_function == 'RAPS': 
#                 # RAPS hyperparameters (currently using ImageNet defaults)
#                 lmbda = .01 
#                 kreg = 5

#                 scores_all = get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True)
#             else:
#                 raise Exception('Undefined score function')

#             for seed in seeds:
#                 print(f'\nseed={seed}')
#                 save_to = os.path.join(curr_folder, f'seed={seed}_allresults.pkl')
#                 if os.path.exists(save_to):
#                     with open(save_to,'rb') as f:
#                         all_results = pickle.load(f)
#                         print('Loaded existing results file containing results for', all_results.keys())
#                 else:
#                     all_results = {} # Each value is (qhat(s), preds, coverage_metrics, set_size_metrics)
                
#                 # Split data
#                 totalcal_scores_all, totalcal_labels, val_scores_all, val_labels = split_X_and_y(scores_all, labels, n_totalcal, num_classes=num_classes, seed=seed)
                
#                 for method in methods:
                    
                    
#                     if method == 'standard':
#                         # Standard conformal
#                         standard_qhat = compute_qhat(totalcal_scores_all, totalcal_labels, alpha=alpha)
#                         standard_preds = create_prediction_sets(val_scores_all, standard_qhat)
#                         coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, standard_preds, alpha)
#                         all_results['standard'] = (standard_qhat, standard_preds, coverage_metrics, set_size_metrics)
                        
#                     elif method == 'classwise':
#                         # Classwise conformal
#                         classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, 
#                                                                        totalcal_labels, alpha=alpha, default_qhat=np.inf)
#                         classwise_preds = create_cb_prediction_sets(val_scores_all, classwise_qhats)

#                         coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
#                         all_results['classwise'] = (classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics)
                        
#                     elif method == 'always_cluster':
#                         # Clustered conformal
#                         all_results['always'] = automatic_clustered_conformal(totalcal_scores_all, totalcal_labels,
#                                                                                         alpha,
#                                                                                 val_scores_all, val_labels, 
#                                                                                 cluster='always')

#                     elif method == 'regularized_classwise':
#                         # Empirical-Bayes-inspired regularized classwise conformal (shrink class qhats to standard)
#                         # TODO
#                         pass

#                 # Optionally remove predictions from saved output to reduce memory usage
#                 if not save_preds:
#                     for m in all_results.keys():
#                         all_results[m] = (all_results[m][0], None, all_results[m][2], all_results[m][3])
                
#                 # Save results 
#                 with open(save_to,'wb') as f:
#                     pickle.dump(all_results, f)
#                     print(f'Saved results to {save_to}')
                    
                    
def initialize_metrics_dict(methods):
    
    metrics = {}
    for method in methods:
        metrics[method] = {'class_cov_gap': [],
                           'max_class_cov_gap': [],
                           'avg_set_size': [],
                           'marginal_cov': [],
                           'very_undercovered': []} # Could also retrieve other metrics
        
    return metrics


def average_results_across_seeds(folder, print_results=True, display_table=True, 
                                 methods=['standard', 'classwise', 'cluster_balanced']):

    
    file_names = sorted(glob.glob(os.path.join(folder, '*.pkl')))
    num_seeds = len(file_names)
#     if display_table:
    print('Number of seeds found:', num_seeds)
    
    metrics = initialize_metrics_dict(methods)
    
    for pth in file_names:
        with open(pth, 'rb') as f:
            results = pickle.load(f)
                        
        for method in methods:
            metrics[method]['class_cov_gap'].append(results[method][2]['mean_class_cov_gap'])
            metrics[method]['avg_set_size'].append(results[method][3]['mean'])
            metrics[method]['max_class_cov_gap'].append(results[method][2]['max_gap'])
            metrics[method]['marginal_cov'].append(results[method][2]['marginal_cov'])
            metrics[method]['very_undercovered'].append(results[method][2]['very_undercovered'])
            
#     # ADDED
#     print(folder)
#     for method in methods:
#         print(method, metrics[method]['class_cov_gap'])
            
    cov_means = []
    cov_ses = []
    set_size_means = []
    set_size_ses = []
    max_cov_gap_means = []
    max_cov_gap_ses = []
    marginal_cov_means = []
    marginal_cov_ses = []
    very_undercovered_means = []
    very_undercovered_ses = []
    
    if print_results:
        print('Avg class coverage gap for each random seed:')
    for method in methods:
        if print_results:
            print(f'  {method}:', np.array(metrics[method]['class_cov_gap'])*100)
        cov_means.append(np.mean(metrics[method]['class_cov_gap']))
        cov_ses.append(np.std(metrics[method]['class_cov_gap']))
        
        set_size_means.append(np.mean(metrics[method]['avg_set_size']))
        set_size_ses.append(np.std(metrics[method]['avg_set_size']))
        
        max_cov_gap_means.append(np.mean(metrics[method]['max_class_cov_gap']))
        max_cov_gap_ses.append(np.std(metrics[method]['max_class_cov_gap']))
        
        marginal_cov_means.append(np.mean(metrics[method]['marginal_cov']))
        marginal_cov_ses.append(np.std(metrics[method]['marginal_cov']))
        
        very_undercovered_means.append(np.mean(metrics[method]['very_undercovered']))
        very_undercovered_ses.append(np.std(metrics[method]['very_undercovered']))
        
    df = pd.DataFrame({'method': methods,
                      'class_cov_gap_mean': np.array(cov_means)*100,
                      'class_cov_gap_se': np.array(cov_ses)*100,
                      'max_class_cov_gap_mean': np.array(max_cov_gap_means)*100,
                      'max_class_cov_gap_se': np.array(max_cov_gap_ses)*100,
                      'avg_set_size_mean': set_size_means,
                      'avg_set_size_se': set_size_ses,
                      'marginal_cov_mean': marginal_cov_means,
                      'marginal_cov_se': marginal_cov_ses,
                      'very_undercovered_mean': very_undercovered_means,
                      'very_undercovered_se': very_undercovered_ses})
    
    if display_table:
        display(df) # For Jupyter notebooks
        
    return df

def plot_class_coverage_histogram(folder, desired_cov=None, vmin=.6, vmax=1, nbins=30, title=None):
    '''
    For each method, aggregate class coverages across all random seeds and then 
    plot density/histogram. This is equivalent to estimating a density for each
    random seed individually then averaging. 
    
    Inputs:
    - folder: (str) containing path to folder of saved results
    - desired_cov: (float) Desired coverage level 
    - vmin, vmax: (floats) Specify bin edges 
    - 
    '''
    sns.set_style(style='white', rc={'axes.spines.right': False, 'axes.spines.top': False})
    sns.set_palette('pastel')
    sns.set_context('talk') # 'paper', 'talk', 'poster'
    
    methods = ['standard', 
               'classwise', 
               'always_cluster']
    
    bin_edges = np.linspace(vmin,vmax,nbins+1)
    
    file_names = sorted(glob.glob(os.path.join(folder, '*.pkl')))
    num_seeds = len(file_names)
    print('Number of seeds found:', num_seeds)
    
    # OPTION 1: Plot average with 95% CIs
    cts_dict = {}
    for method in methods:
        cts_dict[method] = np.zeros((num_seeds, nbins))
        
    for i, pth in enumerate(file_names):
        with open(pth, 'rb') as f:
            results = pickle.load(f)
            
        for method in methods:
            
            cts, _ = np.histogram(results[method][2]['raw_class_coverages'], bins=bin_edges)
            cts_dict[method][i,:] = cts
    
    for method in methods:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        graph = sns.lineplot(x=np.tile(bin_centers, num_seeds), y=np.ndarray.flatten(cts_dict[method]),
                     label=map_to_label[method], color=map_to_color[method])

    if desired_cov is not None:
        graph.axvline(desired_cov, color='black', linestyle='dashed', label='Desired coverage')
        
    plt.xlabel('Class-conditional coverage')
    plt.ylabel('Number of classes')
    plt.title(title)
    plt.ylim(bottom=0)
    plt.xlim(right=vmax)
    plt.legend()
    plt.show()
    
    # OPTION 2: Plot average, no CIs
#     class_coverages = {}
#     for method in methods:
#         class_coverages[method] = []
        
#     for pth in file_names:
#         with open(pth, 'rb') as f:
#             results = pickle.load(f)
            
#         for method in methods:
#             class_coverages[method].append(results[method][2]['raw_class_coverages'])
    
#     bin_edges = np.linspace(vmin,vmax,30) # Can adjust
    
#     for method in methods:
#         aggregated_scores = np.concatenate(class_coverages[method], axis=0)
#         cts, _ = np.histogram(aggregated_scores, bins=bin_edges, density=False)
#         cts = cts / num_seeds 
#         plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, cts, '-o', label=method, alpha=0.7)
        
#     plt.xlabel('Class-conditional coverage')
#     plt.ylabel('Number of classes')
#     plt.legend()

#     # OPTION 3: Plot separate lines
#     class_coverages = {}
#     for method in methods:
#         class_coverages[method] = []
        
#     for pth in file_names:
#         with open(pth, 'rb') as f:
#             results = pickle.load(f)
            
#         for method in methods:
#             class_coverages[method].append(results[method][2]['raw_class_coverages'])
    
#     bin_edges = np.linspace(vmin,vmax,30) # Can adjust
    
#     for method in methods:
#         for class_covs in class_coverages[method]:
#             cts, _ = np.histogram(class_covs, bins=bin_edges, density=False)
#             plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, cts, '-', alpha=0.3,
#                      label=map_to_label[method], color=map_to_color[method])
        
#     plt.xlabel('Class-conditional coverage')
#     plt.ylabel('Number of classes')
#     plt.show()
#     plt.legend()
    