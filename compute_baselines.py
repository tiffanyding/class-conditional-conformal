# Jan 27, 2023

'''
Example command:
    
'''

import time
st = time.time()

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import pickle
import torch

from collections import Counter
from scipy import stats, cluster
from sklearn.cluster import AgglomerativeClustering

from utils.conformal_utils import *


## 0. Parse arguments =====================
parser = argparse.ArgumentParser()
parser.add_argument('softmax_path', type=str, help='File path of numpy array of softmax scores')
parser.add_argument('labels_path', type=str, help='File path of numpy array of labels')
# parser.add_argument('n_min', type=int, help='Minimum number of conformal calibration points per cluster, assuming balanced cluster sizes')
parser.add_argument('--n_totalcal', type=int, default=20,  help='Total number of calibration points (= # clustering examples + # conformal calibration examples')
parser.add_argument("--alpha", type=float, default=.1, help='Desired coverage is set to 1-alpha')

args = parser.parse_args()
# n_min = args.n_min
n_totalcal = args.n_totalcal
alpha = args.alpha

# score_function = 'APS' # 'softmax', 'APS'
# alpha = .1
# n_totalcal = 20 # Total number of calibration points (= # clustering examples + # conformal calibration examples)
# n_min = 100

num_classes = 1000 # Number of classes in ImageNet

print('====== SETTINGS =====')
# print(f'score_function={score_function}')
print(f'alpha={alpha}')
print(f'n_totalcal={n_totalcal}')
# print(f'n_min={n_min}')
print('=====================')
    

## 1. Get data ============================
print('Loading scores and labels...')

softmax_scores = np.load(args.softmax_path)
labels = np.load(args.labels_path)

for score_function in ['softmax', 'APS', 'RAPS']:
    
    print(f'====== score_function={score_function} ======')
    
    print('Computing conformal score...')
    if score_function == 'softmax':
        scores_all = 1 - softmax_scores
    elif score_function == 'APS':
        scores_all = get_APS_scores_all(softmax_scores, randomize=True)
    elif: 
        # RAPS hyperparameters
        lmbda = .01 
        kreg = 5
        
        scores_all = get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True)
    else:
        raise Exception('Undefined score function')


    print('Splitting data...')
    # Split into clustering+calibration data and validation data
    totalcal_scores_all, totalcal_labels, val_scores_all, val_labels = split_X_and_y(scores_all, labels, n_totalcal, num_classes=num_classes, seed=0)


## 2. Compute baselines for evaluation ============================

    print('Evaluating baselines...')
    # A) Vanilla conformal
    vanilla_qhat = compute_qhat(totalcal_scores_all, totalcal_labels, alpha=alpha)
    vanilla_preds = create_prediction_sets(val_scores_all, vanilla_qhat)

    marginal_cov = compute_coverage(val_labels, vanilla_preds)
    print(f'Marginal coverage of Vanilla: {marginal_cov*100:.2f}%')
    vanilla_class_specific_cov = compute_class_specific_coverage(val_labels, vanilla_preds)

    # B) Naive class-balanced
    naivecb_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, alpha=alpha, default_qhat=np.inf)
    naivecb_preds = create_cb_prediction_sets(val_scores_all, naivecb_qhats)

    naivecb_marginal_cov = compute_coverage(val_labels, naivecb_preds)
    print(f'Marginal coverage of NaiveCC: {naivecb_marginal_cov*100:.2f}%')
    naivecb_class_specific_cov = compute_class_specific_coverage(val_labels, naivecb_preds)

    # CC coverage
    vanilla_l1_dist = np.sum(np.abs(vanilla_class_specific_cov - (1 - alpha)))
    naivecb_l1_dist = np.sum(np.abs(naivecb_class_specific_cov - (1 - alpha)))

    print(f'[Vanilla] L1 distance between desired and realized class-cond. coverage: {vanilla_l1_dist:.3f}')
    print(f'[NaiveCC] L1 distance between desired and realized class-cond. coverage: {naivecb_l1_dist:.3f}')
    
    print("Note: The average magnitude of deviation from the desired coverage is L1 dist/1000")

    ## Set size
    vanilla_set_sizes = [len(x) for x in vanilla_preds]
    vanilla_set_size_metrics = {'mean': np.mean(vanilla_set_sizes), '[.25, .5, .75, .9] quantiles': np.quantile(vanilla_set_sizes, [.25, .5, .75, .9])}
    naivecb_set_sizes = [len(x) for x in naivecb_preds]
    naivecb_set_size_metrics = {'mean': np.mean(naivecb_set_sizes), '[.25, .5, .75, .9] quantiles': np.quantile(naivecb_set_sizes, [.25, .5, .75, .9])}
    print(f'[Vanilla] set size metrics:' vanilla_set_size_metrics)
    print(f'[NaiveCC] set size metrics:' naivecb_set_size_metrics)


print(f'TIME TAKEN: {(time.time() - st)/60:.2f} min')

    

