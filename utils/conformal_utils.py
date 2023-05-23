# Copied from empirical-bayes-conformal repo and modified

import matplotlib.pyplot as plt
import numpy as np
import torch

import pdb

from collections import Counter

# For Clustered Conformal
from .clustering_utils import embed_all_classes, test_one_cluster_null
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#========================================
#  Misc.
#========================================

def get_quantile_threshold(alpha):
    '''
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''
    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1
    return n

#========================================
#   Data preparation
#========================================

# Used for creating randomly sampled calibration dataset
def random_split(X, y, avg_num_per_class, seed=0):
    '''
    Randomly splits X and y into X1, y1, X2, y2. X1 and y1 will have of
    avg_num_per_class * num_classes examples. The remaining examples
    will be in X2 and y2.
    
    Inputs:
    - X: numpy array. axis 0 must be the same length as y
    - y: 1-D numpy array of class labels (classes should be 0-indexed)
    - avg_num_per_class: Average number of examples per class to include 
    in the data split. 
    - seed: (int) random seed for reproducibility
    
    Output: X1, y1, X2, y2
    '''

    np.random.seed(seed)
    
    num_classes = np.max(y) + 1
    
    num_samples = avg_num_per_class * num_classes
    
    idx1 = np.random.choice(np.arange(len(y)), size=num_samples, replace=False) # Numeric index
    idx2 = ~np.isin(np.arange(len(y)), idx1) # Boolean index
    X1, y1 = X[idx1], y[idx1]
    X2, y2 = X[idx2], y[idx2]
    
    return X1, y1, X2, y2


# Used for creating balanced or stratified calibration dataset
def split_X_and_y(X, y, n_k, num_classes, seed=0, split='balanced'):
    '''
    Randomly generate two subsets of features X and corresponding labels y such that the
    first subset contains n_k instances of each class k and the second subset contains all
    other instances 
    
    Inputs:
        X: n x d array (e.g., matrix of softmax vectors)
        y: n x 1 array
        n_k: positive int or n x 1 array
        num_classes: total number of classes, corresponding to max(y)
        seed: random seed
        
    Output:
        X1, y1
        X2, y2
    '''
    np.random.seed(seed)
    
    if split == 'balanced':
    
        if not hasattr(n_k, '__iter__'):
            n_k = n_k * np.ones((num_classes,), dtype=int)
    elif split == 'proportional':
        assert not hasattr(n_k, '__iter__')
        
        # Compute what fraction of the rarest class n_clustering corresponds to,
        # then set n_k = frac * (total # of cal points for class k)
        cts = Counter(y)
        rarest_class_ct = cts.most_common()[-1][1]
        frac = n_k / rarest_class_ct
        n_k = [int(frac*cts[k]) for k in range(num_classes)]
        
    else: 
        raise Exception('Valid split options are "balanced" or "proportional"')
            
    
    if len(X.shape) == 2:
        X1 = np.zeros((np.sum(n_k), X.shape[1]))
    else:
        X1 = np.zeros((np.sum(n_k),))
    y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    
    all_selected_indices = np.zeros(y.shape)

    i = 0
    for k in range(num_classes):

        # Randomly select n instances of class k
        idx = np.argwhere(y==k).flatten()
#         pdb.set_trace()
        selected_idx = np.random.choice(idx, replace=False, size=(n_k[k],))

        X1[i:i+n_k[k]] = X[selected_idx]
        y1[i:i+n_k[k]] = k
        i += n_k[k]
        
        all_selected_indices[selected_idx] = 1
        
    X2 = X[all_selected_indices == 0]
    y2 = y[all_selected_indices == 0]
    
    return X1, y1, X2, y2

def get_true_class_conformal_score(scores_all, labels):
    '''
    Extracts conformal scores that corresponds to the true class labels
    
    Inputs:
        scores_all: n x num_classes array 
        labels: length-n array of true class labels
    '''
    return scores_all[np.arange(len(labels)), labels]

#========================================
#   Standard conformal inference
#========================================

def compute_qhat(class_scores, true_labels, alpha, plot_scores=False):
    '''
    Compute quantile q_hat that will result in marginal coverage of (1-alpha)
    
    Inputs:
        class_scores: num_instances x num_classes array of scores, or num_instances-length array of 
        conformal scores for true class. A higher score indicates more uncertainty
        true_labels: num_instances length array of ground truth labels
    
    '''
    # Select scores that correspond to correct label
    if len(class_scores.shape) == 2:
        scores = np.squeeze(np.take_along_axis(class_scores, np.expand_dims(true_labels, axis=1), axis=1))
    else:
        scores = class_scores
    
    # Sort scores
    scores = np.sort(scores)

    # Identify score q_hat such that ~(1-alpha) fraction of scores are below qhat 
    #    Note: More precisely, it is (1-alpha) times a small correction factor
    n = len(true_labels)
    q_hat = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method='inverted_cdf')
    
    # Plot score distribution
    if plot_scores:
        plt.hist(scores)
        plt.title('Score distribution')
        plt.show()

    return q_hat

# Create prediction sets
def create_prediction_sets(class_probs, q_hat):
    assert(not hasattr(q_hat, '__iter__')), "q_hat should be a single number and not a list or array"
    class_probs = np.array(class_probs)
    set_preds = []
    num_samples = len(class_probs)
    for i in range(num_samples):
        set_preds.append(np.where(class_probs[i,:] <= q_hat)[0])
        
    return set_preds

# Standard conformal pipeline
def standard_conformal_pipeline(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha):
    standard_qhat = compute_qhat(totalcal_scores_all, totalcal_labels, alpha=alpha)
    standard_preds = create_prediction_sets(val_scores_all, standard_qhat)
    coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, standard_preds, alpha)
    
    return standard_qhat, standard_preds, coverage_metrics, set_size_metrics

#========================================
#   Class-wise conformal inference and regularized class-wise
#========================================

# Additive version
def reconformalize(qhats, scores, labels, alpha, adjustment_min=-1, adjustment_max=1):
    '''
    Adjust qhats by additive factor so that marginal coverage of 1-alpha is achieved
    '''
    print('Applying additive adjustment to qhats')
    # ===== Perform binary search =====
    # Convergence criteria: Either (1) marginal coverage is within tol of desired or (2)
    # quantile_min and quantile_max differ by less than .001, so there is no need to try 
    # to get a more precise estimate
    tol = 0.0005

    marginal_coverage = 0
    while np.abs(marginal_coverage - (1-alpha)) > tol:

        adjustment_guess = (adjustment_min +  adjustment_max) / 2
        print(f"\nCurrent adjustment: {adjustment_guess:.6f}")

        curr_qhats = qhats + adjustment_guess 

        preds = create_cb_prediction_sets(scores, curr_qhats)
        marginal_coverage = compute_coverage(labels, preds)
        print(f"Marginal coverage: {marginal_coverage:.4f}")

        if marginal_coverage > 1 - alpha:
            adjustment_max = adjustment_guess
        else:
            adjustment_min = adjustment_guess
        print(f"Search range: [{adjustment_min}, {adjustment_max}]")

        if adjustment_max - adjustment_min < .00001:
            adjustment_guess = adjustment_max # Conservative estimate, which ensures coverage
            print("Adequate precision reached; stopping early.")
            break
            
    print('Final adjustment:', adjustment_guess)
    qhats += adjustment_guess
    
    return qhats

# # Multiplicative version
# def reconformalize(qhats, scores, labels, alpha, adjustment_min=0, adjustment_max=2):
#     '''
#     Adjust qhats by multiplicative factor so that marginal coverage of 1-alpha is achieved
#     '''
#     print('Applying multiplicative adjustment to qhats')
#     # ===== Perform binary search =====
#     # Convergence criteria: Either (1) marginal coverage is within tol of desired or (2)
#     # quantile_min and quantile_max differ by less than .001, so there is no need to try 
#     # to get a more precise estimate
#     tol = 0.0005

#     marginal_coverage = 0
#     while np.abs(marginal_coverage - (1-alpha)) > tol:

#         adjustment_guess = (adjustment_min +  adjustment_max) / 2
#         print(f"\nCurrent adjustment: {adjustment_guess:.6f}")

#         curr_qhats = qhats * adjustment_guess 

#         preds = create_cb_prediction_sets(scores, curr_qhats)
#         marginal_coverage = compute_coverage(labels, preds)
#         print(f"Marginal coverage: {marginal_coverage:.4f}")

#         if marginal_coverage > 1 - alpha:
#             adjustment_max = adjustment_guess
#         else:
#             adjustment_min = adjustment_guess
#         print(f"Search range: [{adjustment_min}, {adjustment_max}]")

#         if adjustment_max - adjustment_min < .00001:
#             adjustment_guess = adjustment_max # Conservative estimate, which ensures coverage
#             print("Adequate precision reached; stopping early.")
#             break
            
#     print('Final adjustment:', adjustment_guess)
#     qhats += adjustment_guess
    
#     return qhats   

def compute_class_specific_qhats(cal_class_scores, cal_true_labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_qhat='standard', regularize=False):
    '''
    Computes class-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - cal_class_scores: 
            num_instances x num_classes array where cal_class_scores[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed). If class -1 appears,
        it will be assigned the default_qhat value. It is appended as an extra entry of the returned q_hats
        so that q_hats[-1] = null_qhat
        - alpha: Determines desired coverage level
        - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class 
        specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
        conformal and use that as the default value
        - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
        cal_true_labels. null_qhat is assigned to 
        class/cluster -1 
        - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
        shrinkage for a class is determined by number of samples of that class
    '''
    
    if default_qhat == 'standard':
        default_qhat = compute_qhat(cal_class_scores, cal_true_labels, alpha=alpha)
        
    if null_qhat == 'standard':
        null_qhat = compute_qhat(cal_class_scores, cal_true_labels, alpha=alpha)
     
    # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats 
    # to get a marginal coverage guarantee
    if regularize:
        num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
        idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
        idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
        reserved_scores, reserved_labels = cal_class_scores[idx], cal_true_labels[idx]
        cal_class_scores, cal_true_labels = cal_class_scores[idx2], cal_true_labels[idx2]
                 
    num_samples = len(cal_true_labels)
    q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
    class_cts = np.zeros((num_classes,))
    for k in range(num_classes):

        # Only select data for which k is true class
        idx = (cal_true_labels == k)

        if len(cal_class_scores.shape) == 2:
            scores = cal_class_scores[idx, k]
        else:
            scores = cal_class_scores[idx]

        class_cts[k] = scores.shape[0]

        if len(scores) == 0:
            assert default_qhat is not None, f"Class/cluster {k} does not appear in the calibration set, so the quantile for this class cannot be computed. Please specify a value for default_qhat to use in this case."
            print(f'Warning: Class/cluster {k} does not appear in the calibration set,', 
                  f'so default q_hat value of {default_qhat} will be used')
            q_hats[k] = default_qhat
        else:
            scores = np.sort(scores)
            num_samples = len(scores)
            val = np.ceil((num_samples+1)*(1-alpha)) / num_samples
            if val > 1:
                assert default_qhat is not None, f"Class/cluster {k} does not appear enough times to compute a proper quantile. Please specify a value for default_qhat to use in this case."
                print(f'Warning: Class/cluster {k} does not appear enough times to compute a proper quantile,', 
                      f'so default q_hat value of {default_qhat} will be used')
                q_hats[k] = default_qhat
            else:
                q_hats[k] = np.quantile(scores, val, method='inverted_cdf')
                
    if -1 in cal_true_labels:
        q_hats = np.concatenate((q_hats, [null_qhat]))
     
    # Optionally apply shrinkage 
    if regularize:
        N = num_classes
        n_k = np.maximum(class_cts, 1) # So that classes that never appear do not cause division by 0 issues. 
        shrinkage_factor = .03 * n_k # smaller = less shrinkage
        shrinkage_factor = np.minimum(shrinkage_factor, 1)
        print('SHRINKAGE FACTOR:', shrinkage_factor)  
        print(np.min(shrinkage_factor), np.max(shrinkage_factor))
        q_hats = default_qhat + shrinkage_factor * (q_hats - default_qhat)
        
        # Correct qhats via additive factor to achieve marginal coverage
        q_hats = reconformalize(q_hats, reserved_scores, reserved_labels, alpha)

       
    return q_hats

# Create class_balanced prediction sets
def create_cb_prediction_sets(class_scores, q_hats):
    '''
    Inputs:
        - class_scores: num_instances x num_classes array where class_scores[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
    '''
    class_scores = np.array(class_scores)
    set_preds = []
    num_samples = len(class_scores)
    for i in range(num_samples):
        set_preds.append(np.where(class_scores[i,:] <= q_hats)[0])
        
    return set_preds


# Classwise conformal pipeline
def classwise_conformal_pipeline(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha,
                                 num_classes,
                                 default_qhat=np.inf, regularize=False):
    '''
    See compute_class_specific_qhats() for doc-string
    
    '''
    
    classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize)
    classwise_preds = create_cb_prediction_sets(val_scores_all, classwise_qhats)

    coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics
                

#========================================
#   Clustered conformal prediction
#========================================

def compute_cluster_specific_qhats(cluster_assignments, cal_class_scores, cal_true_labels, alpha, null_qhat='standard'):
    '''
    Computes cluster-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - cluster_assignments: num_classes length array where entry i is the index of the cluster that class i belongs to.
          Clusters should be 0-indexed. Rare classes can be assigned to cluster -1 and they will automatically be given
          qhat_k = default_qhat. 
        - cal_class_scores: num_instances x num_classes array where class_scores[i,j] = score of class j for instance i.
         Alternatively, a num_instances-length array of conformal scores for true class
        - cal_true_labels: num_instances length array of true class labels (0-indexed)
        - alpha: Determines desired coverage level
        - null_qhat: For classes that do not appear in cal_true_labels, the class specific qhat is set to null_qhat.
        If null_qhat == 'standard', we compute the qhat for standard conformal and use that as the default value
    Output:
        num_classes length array where entry i is the quantile correspond to the cluster that class i belongs to. 
        All classes in the same cluster have the same quantile.
    '''
    # If we want the null_qhat to be the standard qhat, we should compute this before we remap the values
    if null_qhat == 'standard':
        null_qhat = compute_qhat(cal_class_scores, cal_true_labels, alpha)
        
    # Edge case: all cluster_assignments are -1. 
    if np.all(cluster_assignments==-1):
        return null_qhat * np.ones(cluster_assignments.shape)
    
    # Extract conformal scores for true labels if not already done
    if len(cal_class_scores) == 2:
        cal_class_scores = cal_class_scores[np.arange(len(cal_true_labels)), cal_true_labels]
    
    # Map true class labels to clusters
    cal_true_clusters = np.array([cluster_assignments[label] for label in cal_true_labels])
    
    # Compute cluster qhats
    cluster_qhats = compute_class_specific_qhats(cal_class_scores, cal_true_clusters, 
                                                 alpha=alpha, num_classes=np.max(cluster_assignments)+1,
                                                 default_qhat=np.inf,
                                                 null_qhat='standard')                            
    # Map cluster qhats back to classes
    num_classes = len(cluster_assignments)
    class_qhats = np.array([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)])
    
    return class_qhats

    # Note: To create prediction sets, just pass class_qhats into create_cb_prediction_sets()
    
    
def clustered_conformal(totalcal_scores_all, totalcal_labels,
                        alpha,
                        val_scores_all=None, val_labels=None,
                        frac_clustering='auto', num_clusters='auto',
                        split='random'):
    '''
    Use totalcal_scores and total_labels to compute conformal quantiles for each
    class using the clustered conformal procedure. Optionally evaluates 
    performance on val_scores and val_labels
    
    Clustered conformal procedure:
        1. Filter out classes where np.ceil((num_samples+1)*(1-alpha)) / num_samples < 1. Assign those classes the standard qhat
        2. To select n_clustering and num_clusters, apply heuristic (which takes as input num_samples and num_classes) to num_samples of rarest remaining class
        3. Run clustering with chosen hyperparameters. Estimate a qhat for each cluster.
    
    Inputs:
         - totalcal_scores: num_instances x num_classes array where 
           cal_class_scores[i,j] = score of class j for instance i
         - totalcal_labels: num_instances-length array of true class labels (0-indexed classes)
         - alpha: number between 0 and 1 that determines coverage level.
         Coverage level will be 1-alpha.
         - n_clustering: Number of points per class to use for clustering step. The remaining
         points are used for the conformal calibration step.
         - num_clusters: Number of clusters to group classes into
         - val_scores: num_val_instances x num_classes array, or None. If not None, 
         the class coverage gap and average set sizes will be computed on val_scores
         and val_labels.
         - val_labels: num_val_instances-length array of true class labels, or None. 
         If not None, the class coverage gap and average set sizes will be computed 
         on val_scores and val_labels.
         - split: How to split data between clustering step and calibration step. Options are
         'balanced' (sample n_clustering per class), 'proportional' (sample proportional to
         distribution such that rarest class has n_clustering example), 'doubledip' (don't
         split and use all data for both steps, or 'random' (each example is assigned to
         clustering step with some fixed probability) 
         
    Outputs:
        - qhats: num_classes-length array where qhats[i] = conformal quantial estimate for class i
        - [Optionally, if val_scores and val_labels are not None] 
            - val_preds: clustered conformal predictions on val_scores
            - val_class_coverage_gap: Class coverage gap, compute on val_scores and val_labels
            - val_set_size_metrics: Dict containing set size metrics, compute on val_scores and val_labels
            
    '''
    def get_rare_classes(labels, alpha, num_classes):
        thresh = get_quantile_threshold(alpha)
        classes, cts = np.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh]
        
        # Also included any classes that are so rare that we have 0 labels for it
        zero_ct_classes = np.setdiff1d(np.arange(num_classes), classes)
        rare_classes = np.concatenate((rare_classes, zero_ct_classes))
        
        return rare_classes
        
    def remap_classes(labels, rare_classes):
        '''
        Exclude classes in rare_classes and remap remaining classes to be 0-indexed

        Outputs:
            - remaining_idx: Boolean array the same length as labels. Entry i is True
            iff labels[i] is not in rare_classes 
            - remapped_labels: Array that only contains the entries of labels that are 
            not in rare_classes (in order) 
            - remapping: Dict mapping old class index to new class index

        '''
        remaining_idx = ~np.isin(labels, rare_classes)

        remaining_labels = labels[remaining_idx]
        remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
        new_idx = 0
        remapping = {}
        for i in range(len(remaining_labels)):
            if remaining_labels[i] in remapping:
                remapped_labels[i] = remapping[remaining_labels[i]]
            else:
                remapped_labels[i] = new_idx
                remapping[remaining_labels[i]] = new_idx
                new_idx += 1
        return remaining_idx, remapped_labels, remapping
    
    # Data preperation: Get conformal scores for true classes
    num_classes = totalcal_scores_all.shape[1]
    totalcal_scores = get_true_class_conformal_score(totalcal_scores_all, totalcal_labels)
    
    # 1) Apply heuristic to choose hyperparameters if not prespecified
    if frac_clustering == 'auto' and num_clusters == 'auto':
        cts_dict = Counter(totalcal_labels)
        cts = [cts_dict.get(k, 0) for k in range(num_classes)]
        n_min = min(cts)
        n_thresh = get_quantile_threshold(alpha) 
        n_min = max(n_min, n_thresh) # Classes with fewer than n_thresh examples will be excluded from clustering
        num_remaining_classes = np.sum(np.array(list(cts)) >= n_min)


    #     # 1) Identify rare classes
    #     rare_classes = get_rare_classes(totalcal_labels, alpha)
    #     n_thresh = get_quantile_threshold(alpha) 
    #     print(f'Excluding {len(rare_classes)} rare classes (out of {num_classes}) from clustering')

    #     # 2) Apply heuristic to choose hyperparameters
    #     num_remaining_classes = num_classes - len(rare_classes) 
    #     filtered_labels = totalcal_labels[~np.isin(totalcal_labels, rare_classes)]
    #     n_min = min(Counter(filtered_labels).values())
    #     n_min = max(n_min, n_thresh) # Classes with fewer than n_thresh examples will be excluded

        n_clustering, num_clusters = get_clustering_parameters(num_remaining_classes, n_min)
        print(f'n_clustering={n_clustering}, num_clusters={num_clusters}')
        # Convert n_clustering to fraction relative to n_min
        frac_clustering = n_clustering / n_min

        
    # 2a) Split data
    if split == 'proportional':
        n_k = [int(frac_clustering*cts[k]) for k in range(num_classes)]
        scores1, labels1, scores2, labels2 = split_X_and_y(totalcal_scores, 
                                                           totalcal_labels, 
                                                           n_k, 
                                                           num_classes=num_classes, 
                                                           seed=0)
#                                                            split=split, # Balanced or stratified sampling 
    elif split == 'doubledip':
        scores1, labels1 = totalcal_scores, totalcal_labels
        scores2, labels2 = totalcal_scores, totalcal_labels
    elif split == 'random':
        # Each point is assigned to clustering set w.p. frac_clustering 
        idx1 = np.random.uniform(size=(len(totalcal_labels),)) < frac_clustering 
        scores1 = totalcal_scores[idx1]
        labels1 = totalcal_labels[idx1]
        scores2 = totalcal_scores[~idx1]
        labels2 = totalcal_labels[~idx1]
        
    else:
        raise Exception('Invalid split. Options are balanced, proportional, doubledip, and random')

    
    # 2b)  Identify "rare" classes = classes that have fewer than 1/alpha - 1 examples 
    # in the clustering set 
    rare_classes = get_rare_classes(labels1, alpha, num_classes)
    print(f'{len(rare_classes)} of {num_classes} classes are rare in the clustering set'
          ' and will be assigned to the null cluster')
    
    # 3) Run clustering
    if num_classes - len(rare_classes) > num_clusters and num_clusters > 1:  
        # Filter out rare classes and re-index
        remaining_idx, filtered_labels, class_remapping = remap_classes(labels1, rare_classes)
        filtered_scores = scores1[remaining_idx]
        
        # Compute embedding for each class and get class counts
        embeddings, class_cts = embed_all_classes(filtered_scores, filtered_labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True)
    
        kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10).fit(embeddings, sample_weight=np.sqrt(class_cts))
        nonrare_class_cluster_assignments = kmeans.labels_  

        # Print cluster sizes
        print(f'Cluster sizes:', [x[1] for x in Counter(nonrare_class_cluster_assignments).most_common()])

        # Remap cluster assignments to original classes. Any class not included in kmeans clustering is a rare 
        # class, so we will assign it to cluster "-1" = num_clusters by Python indexing
        cluster_assignments = -np.ones((num_classes,), dtype=int)
        for cls, remapped_cls in class_remapping.items():
            cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
    else: 
        cluster_assignments = -np.ones((num_classes,), dtype=int)
        print('Skipped clustering because the number of clusters requested was <= 1')
        
    # 4) Compute qhats for each cluster
    cal_scores_all = scores2
    cal_labels = labels2
    qhats = compute_cluster_specific_qhats(cluster_assignments, 
               cal_scores_all, cal_labels, 
               alpha=alpha, 
               null_qhat='standard')

    # 5) [Optionally] Apply to val set. Evaluate class coverage gap and set size 
    if (val_scores_all is not None) and (val_labels is not None):
        preds = create_cb_prediction_sets(val_scores_all, qhats)
        class_cov_metrics, set_size_metrics = compute_all_metrics(val_labels, preds, alpha,
                                                                  cluster_assignments=cluster_assignments)

        # Add # of classes excluded from clustering to class_cov_metrics
        class_cov_metrics['num_unclustered_classes'] = len(rare_classes)
        return qhats, preds, class_cov_metrics, set_size_metrics
    else:
        return qhats

    
def get_clustering_parameters(num_classes, n_totalcal):
    '''
    Returns a guess of good values for num_clusters and n_clustering based solely 
    on the number of classes and the number of examples per class. 
    
    This relies on two heuristics:
    1) We want at least 150 points per cluster on average
    2) We need more samples as we try to distinguish between more distributions. 
    To distinguish between 2 distribution, want at least 4 samples per class. 
    To distinguish between 5 distributions, want at least 10 samples per class. 
    
    Output: n_clustering, num_clusters
    
    '''
    # Alias for convenience
    K = num_classes
    N = n_totalcal
    
    n_clustering = int(N*K/(75+K))
    num_clusters = int(np.floor(n_clustering / 2))
    
    return n_clustering, num_clusters

#========================================
#   Conformal variant: exact coverage via randomization 
#========================================
# Currently implemented for classwise only, but can also be used for standard conformal or clustered conformal

def get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf):
    '''
    Compute the quantities necessary for performing conformal prediction with exact coverage
    
    Inputs:
        score: length-n array of conformal scores
        alpha: float between 0 and 1 denoting the desired coverage level
        default_qhat = value used when n is too small for computing the relevant quantiles
    Outputs:
        q_a = The smallest score that is larger than (n+1) * (1-alpha) of the other scores (= normal conformal qhat)
        q_b The smallest score that is larger than (n+1) * (1-alpha) - 1 of the other scores
        gamma: value between 0 and 1 such that gamma * q_a + (1-gamma)*g_b = 1-alpha
    '''
    n = len(scores)
    
    if n == 0:
        return np.inf, np.inf, 1
    
    val_a = np.ceil((n+1)*(1-alpha)) / n
    if val_a > 1:
        q_a = default_qhat
    else:
        q_a = np.quantile(scores, val_a, method='inverted_cdf')
        
    val_b = (np.ceil((n+1)*(1-alpha)-1)) / n
    if val_b > 1:
        q_b = default_qhat
        
    else:
        q_b = np.quantile(scores, val_b, method='inverted_cdf') 
        
    if val_a > 1 and val_b > 1:
        gamma = 1 # Any value would work, since q_a and q_b are both equal to default_qhat
    else:
        overcov = np.ceil((n+1)*(1-alpha))/(n+1) - (1-alpha) # How much coverage using q_a will exceed 1 - alpha
        undercov = (1-alpha) - (np.ceil((n+1)*(1-alpha))-1)/(n+1)  #  How much coverage using q_b will undershoot 1 - alpha
#         print(f'overcov: {overcov:.3f}, undercov: {undercov:.3f}')
        gamma = undercov / (undercov + overcov)

    return q_a, q_b, gamma

def construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, class_scores):
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
    
    
    class_scores = np.array(class_scores)
    set_preds = []
    num_samples = len(class_scores)
    for i in range(num_samples):
        try:
            Bs = np.random.rand(len(gammas)) < gammas # Bernoulli random vars
#             val1 = np.multiply(q_as, Bs, where=(~(np.isinf(q_as) & (Bs==0)))) # Define 0 * inf = 0
#             val2 = np.multiply(q_bs, 1-Bs, where=(~(np.isinf(q_as) & ((1-Bs)==0))))
#             q_hats = val1 + val2
            q_hats = np.where(Bs, q_as, q_bs)
            set_preds.append(np.where(class_scores[i,:] <= q_hats)[0]) 
        except:
            pdb.set_trace()
        
    return set_preds

def exact_coverage_classwise_conformal_pipeline(scores_all, labels, num_classes, alpha, default_qhat=np.inf, 
                                       val_scores_all=None, val_labels=None):
    num_classes = scores_all.shape[1]
    
    q_as = np.zeros((num_classes,))   
    q_bs = np.zeros((num_classes,)) 
    gammas = np.zeros((num_classes,)) 
    for k in range(num_classes):
        # Only select data for which k is true class
        idx = (labels == k)
        scores = scores_all[idx, k]
        
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf)
        q_as[k] = q_a
        q_bs[k] = q_b
        gammas[k] = gamma
        
    exact_cov_params = {'q_a': q_as, 'q_b': q_bs, 'gamma': gammas}
    
    # Optionally, compute coverage metrics on validation data
    if (val_scores_all is not None) and (val_labels is not None):
        preds = construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, val_scores_all)
        class_cov_metrics, set_size_metrics = compute_all_metrics(val_labels, preds, alpha)

        return exact_cov_params, preds, class_cov_metrics, set_size_metrics
    else:
        return exact_cov_params
        
#========================================
#   Adaptive Prediction Sets (APS)
#========================================

def get_APS_scores(softmax_scores, labels, randomize=True):
    '''
    Compute conformity score defined in Romano et al, 2020
    (Including randomization, unless randomize is set to False)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
    
    Output: 
        length-n array of APS scores
    '''
    n = len(labels)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
    
def get_APS_scores_all(softmax_scores, randomize=True):
    '''
    Similar to get_APS_scores(), except the APS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores
    else:
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores

#========================================
#   Regularized Adaptive Prediction Sets (RAPS)
#========================================

def get_RAPS_scores(softmax_scores, labels, lmbda, kreg, randomize=True):
    '''
    Essentially the same as get_APS_scores() except with regularization
    
    '''
    n = len(labels)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    
    # Regularization
    y_rank = pi.argsort(1)[range(labels_calib.shape[0]), labels_calib] + 1 # Compute softmax rank of true labels y
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=y_rank.shape))
    scores += reg
    
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
        
def get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True):
    '''
    Similar to get_RAPS_scores(), except the RAPS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    
    # Regularization (pretend each class is true label)
    y_rank = pi.argsort(1) + 1 # Compute softmax rank of true labels y
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=scores.shape))
 
    scores += reg
        
    if not randomize:
        return scores - softmax_scores
    else:
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores

    
#========================================
#   Evaluation
#========================================

# Helper function for computing accuracy (marginal coverage) of confidence sets
def compute_coverage(true_labels, set_preds):
    true_labels = np.array(true_labels) # Convert to numpy to avoid weird pytorch tensor issues
    num_correct = 0
    for true_label, preds in zip(true_labels, set_preds):
        if true_label in preds:
            num_correct += 1
    set_pred_acc = num_correct / len(true_labels)
    
    return set_pred_acc

# Helper function for computing class-specific coverage of confidence sets
def compute_class_specific_coverage(true_labels, set_preds):
    num_classes = max(true_labels) + 1
    class_specific_cov = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = np.where(true_labels == k)[0]
        selected_preds = [set_preds[i] for i in idx]
        num_correct = np.sum([1 if np.any(pred_set == k) else 0 for pred_set in selected_preds])
        class_specific_cov[k] = num_correct / len(selected_preds)
        
    return class_specific_cov

# Helper function for computing average set size
def compute_avg_set_size(list_of_arrays):
    return np.mean([len(arr) for arr in list_of_arrays])

def compute_all_metrics(val_labels, preds, alpha, cluster_assignments=None):
    class_cond_cov = compute_class_specific_coverage(val_labels, preds)
        
    # Average class coverage gap
    avg_class_cov_gap = np.mean(np.abs(class_cond_cov - (1-alpha)))

    # Average gap for classes that are over-covered
    overcov_idx = (class_cond_cov > (1-alpha))
    overcov_gap = np.mean(class_cond_cov[overcov_idx] - (1-alpha))

    # Average gap for classes that are under-covered
    undercov_idx = (class_cond_cov < (1-alpha))
    undercov_gap = np.mean(np.abs(class_cond_cov[undercov_idx] - (1-alpha)))
    
    # Fraction of classes that are at least 10% under-covered
    thresh = .1
    very_undercovered = np.mean(class_cond_cov < (1-alpha-thresh))
    
    # Max gap
    max_gap = np.max(np.abs(class_cond_cov - (1-alpha)))

    # Marginal coverage
    marginal_cov = compute_coverage(val_labels, preds)

    class_cov_metrics = {'mean_class_cov_gap': avg_class_cov_gap, 
                         'undercov_gap': undercov_gap, 
                         'overcov_gap': overcov_gap, 
                         'max_gap': max_gap,
                         'very_undercovered': very_undercovered,
                         'marginal_cov': marginal_cov,
                         'raw_class_coverages': class_cond_cov,
                         'cluster_assignments': cluster_assignments # Also save class cluster assignments
                        }

    curr_set_sizes = [len(x) for x in preds]
    set_size_metrics = {'mean': np.mean(curr_set_sizes), '[.25, .5, .75, .9] quantiles': np.quantile(curr_set_sizes, [.25, .5, .75, .9])}
    
    print('CLASS COVERAGE GAP:', avg_class_cov_gap)
    print('AVERAGE SET SIZE:', np.mean(curr_set_sizes))
    
    return class_cov_metrics, set_size_metrics