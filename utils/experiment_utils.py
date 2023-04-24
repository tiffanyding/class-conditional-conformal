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

def run_experiment(softmax_scores, labels,
                  save_folder,
                  alpha=.1,
                  n_totalcal_list=[10, 30],
                  score_function_list = ['softmax', 'APS'],
                  seeds = [0,1,2,3,4],
                  save_preds=False):
    '''
    If save_preds is True, the val prediction sets are included in the saved outputs
    '''
    
    num_classes = softmax_scores.shape[1]
    
    for n_totalcal in n_totalcal_list:
        for score_function in score_function_list:
            curr_folder = os.path.join(save_folder, f'n_totalcal={n_totalcal}/score={score_function}')
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
                
                # Split data
                totalcal_scores_all, totalcal_labels, val_scores_all, val_labels = split_X_and_y(scores_all, labels, n_totalcal, num_classes=num_classes, seed=seed)
    
                # 1) Compute baselines
                # Standard conformal
                standard_qhat = compute_qhat(totalcal_scores_all, totalcal_labels, alpha=alpha)
                standard_preds = create_prediction_sets(val_scores_all, standard_qhat)
                
                coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, standard_preds, alpha)
                standard_results = (standard_qhat, standard_preds, coverage_metrics, set_size_metrics)
                
                # Class-wise conformal
                classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, alpha=alpha, default_qhat=np.inf)
                classwise_preds = create_cb_prediction_sets(val_scores_all, classwise_qhats)
                
                coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
                classwise_results = (classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics)

                # 2) Always Cluster
                # results contain qhats, preds, coverage_metrics, set_size_metrics
                always_cluster_results = automatic_clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                                alpha,
                                                                                val_scores_all, val_labels, 
                                                                                cluster='always')

                # 3) Smart Cluster
                smart_cluster_results = automatic_clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                                alpha,
                                                                                val_scores_all, val_labels, 
                                                                                cluster='smart')

                # Save results 
                all_results = {'standard': standard_results,
                               'classwise': classwise_results,
                               'always_cluster': always_cluster_results,
                               'smart_cluster': smart_cluster_results}
                
                # Optionally remove predictions from saved output to reduce memory usage
                if not save_preds:
                    for m in all_results.keys():
                        all_results[m] = (all_results[m][0], None, all_results[m][2], all_results[m][3])
                
                with open(save_to,'wb') as f:
                    pickle.dump(all_results, f)
                    print(f'Saved results to {save_to}')
                    
                    
def initialize_metrics_dict(methods):
    
    metrics = {}
    for method in methods:
        metrics[method] = {'class_cov_gap': [],
                           'max_class_cov_gap': [],
                           'avg_set_size': []} # Could also retrieve other metrics
        
    return metrics


def average_results_across_seeds(folder, print_results=True):
    
    methods = ['standard', 
               'classwise', 
#                'smart_cluster', 
               'always_cluster']
    
    file_names = sorted(glob.glob(os.path.join(folder, '*.pkl')))
    num_seeds = len(file_names)
    print('Number of seeds found:', num_seeds)
    
    metrics = initialize_metrics_dict(methods)
    
    for pth in file_names:
        with open(pth, 'rb') as f:
            results = pickle.load(f)
            
        for method in methods:
            metrics[method]['class_cov_gap'].append(results[method][2]['mean_class_cov_gap'])
            metrics[method]['avg_set_size'].append(results[method][3]['mean'])
            metrics[method]['max_class_cov_gap'].append(results[method][2]['max_gap'])
            
    cov_means = []
    cov_ses = []
    set_size_means = []
    set_size_ses = []
    max_cov_gap_means = []
    max_cov_gap_ses = []
    
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
        
    df = pd.DataFrame({'method': methods,
                      'class_cov_gap_mean': np.array(cov_means)*100,
                      'class_cov_gap_se': np.array(cov_ses)*100,
                      'max_class_cov_gap_mean': np.array(max_cov_gap_means)*100,
                      'max_class_cov_gap_se': np.array(max_cov_gap_ses)*100,
                      'avg_set_size_mean': set_size_means,
                      'avg_set_size_se': set_size_ses})
    
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
    