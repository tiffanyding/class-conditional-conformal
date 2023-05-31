#!/usr/bin/env python
# coding: utf-8

# In[13]:

import warnings
warnings.simplefilter('error')


from utils.clustering_utils import *
from utils.conformal_utils import *
from utils.experiment_utils import *

# Test randomized classwise for achieving exact coverage

# Load data
softmax_scores, labels = load_dataset('imagenet')
scores_all = 1 - softmax_scores
num_classes = softmax_scores.shape[1]

# Settings
alpha = 0.1
avg_num_per_class = 20


# Compute long-run average raw_class_coverage
num_trials = 3
unrand_class_covs = np.zeros((num_trials, num_classes))
rand_class_covs = np.zeros((num_trials, num_classes))
clust_rand_class_covs = np.zeros((num_trials, num_classes))

for i in range(num_trials):
    # Split data
    scores_all1, labels1, scores_all2, labels2 = random_split(scores_all, labels, avg_num_per_class, seed=i)
    
    # Unrandomized classwise
    classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics = classwise_conformal_pipeline(scores_all1, labels1, 
                                                                                                    scores_all2, labels2, 
                                                                                                    alpha,
                                                                                                    num_classes,
                                                                                                    default_qhat=np.inf, 
                                                                                                        regularize=False)
    unrand_class_covs[i,:] = coverage_metrics['raw_class_coverages']
    
    # Randomized classwise
    exact_cov_params, preds, class_cov_metrics, set_size_metrics2 = exact_coverage_classwise_conformal_pipeline(scores_all1, labels1, num_classes, alpha, default_qhat=np.inf, 
                                       val_scores_all=scores_all2, val_labels=labels2)
    rand_class_covs[i,:] = class_cov_metrics['raw_class_coverages']
    
    # Randomized clustered
    exact_cov_params, preds, class_cov_metrics, set_size_metrics2 = clustered_conformal(scores_all1, labels1,
                        alpha,
                        val_scores_all=scores_all2, val_labels=labels2,
                        frac_clustering='auto', num_clusters='auto',
                        split='random',
                        exact_coverage=True)
    clust_rand_class_covs[i,:] = class_cov_metrics['raw_class_coverages']
    

print('UNRAND CLASSWISE', np.mean(unrand_class_covs, axis=0))


print('RAND CLASSWISE', np.mean(rand_class_covs, axis=0))

print('RAND CLUSTERED', np.mean(clust_rand_class_covs, axis=0))

print('UNRAND CLASSWISE', np.mean(unrand_class_covs, axis=1))


print('RAND CLASSWISE', np.mean(rand_class_covs, axis=1))

print('RAND CLUSTERED', np.mean(clust_rand_class_covs, axis=1))

