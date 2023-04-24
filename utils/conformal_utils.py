# Copied from empirical-bayes-conformal repo and modified

import matplotlib.pyplot as plt
import numpy as np
import torch

from collections import Counter

# For Clustered Conformal
from .clustering_utils import embed_all_classes, test_one_cluster_null
from sklearn.cluster import KMeans

#========================================
#   Data preparation
#========================================

def split_X_and_y(X, y, n_k, num_classes=1000, seed=0):
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
    
    if not hasattr(n_k, '__iter__'):
        n_k = n_k * np.ones((num_classes,), dtype=int)
    
    X1 = np.zeros((np.sum(n_k), X.shape[1]))
    y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    
    all_selected_indices = np.zeros(y.shape)

    i = 0
    for k in range(num_classes):

        # Randomly select n instances of class k
        idx = np.argwhere(y==k).flatten()
        selected_idx = np.random.choice(idx, replace=False, size=(n_k[k],))

        X1[i:i+n_k[k], :] = X[selected_idx, :]
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

def compute_qhat(class_scores, true_labels, alpha=.05, plot_scores=False):
    '''
    Compute quantile q_hat that will result in marginal coverage of (1-alpha)
    
    Inputs:
        class_scores: num_instances x num_instances array of scores, where a higher score indicates more uncertainty
        true_labels: num_instances length array of ground truth labels
    
    '''
    # Select scores that correspond to correct label
    scores = np.squeeze(np.take_along_axis(class_scores, np.expand_dims(true_labels, axis=1), axis=1))
    
    # Sort scores
    scores = np.sort(scores)

    # Identify score q_hat such that ~(1-alpha) fraction of scores are below qhat 
    #    Note: More precisely, it is (1-alpha) times a small correction factor
    n = len(true_labels)
    q_hat = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n)
    
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

#========================================
#   (Naive) Class-balanced conformal inference
#========================================

def compute_class_specific_qhats(cal_class_scores, cal_true_labels, alpha=.05, default_qhat=None):
    '''
    Computes class-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - cal_class_scores: 
            num_instances x num_classes array where cal_class_scores[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed)
        - alpha: Determines desired coverage level
        - default_qhat: For classes that do not appear in cal_true_labels, the class specific qhat is set to default_qhat
    '''
    num_samples = len(cal_true_labels)
    num_classes = np.max(cal_true_labels) + 1
    q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
    for k in range(num_classes):
        
        # Only select data for which k is true class
        idx = (cal_true_labels == k)
        
        if len(cal_class_scores.shape) == 2:
            scores = cal_class_scores[idx, k]
        else:
            scores = cal_class_scores[idx]
        
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
#                 q_hats[k] = np.inf
            else:
                q_hats[k] = np.quantile(scores, val)
       
#     print('q_hats', q_hats)
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

#========================================
#   Clustered conformal prediction
#========================================

def compute_cluster_specific_qhats(cluster_assignments, cal_class_scores, cal_true_labels, alpha=.05, default_qhat=None):
    '''
    Computes cluster-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - cluster_assignments: num_classes length array where entry i is the index of the cluster that class i belongs to.
          Clusters should be 0-indexed.
        - cal_class_scores: num_instances x num_classes array where class_scores[i,j] = score of class j for instance i
        - cal_true_labels: num_instances length array of true class labels (0-indexed)
        - alpha: Determines desired coverage level
        - default_qhat: For classes that do not appear in cal_true_labels, the class specific qhat is set to default_qhat
        
    Output:
        num_classes length array where entry i is the quantile correspond to the cluster that class i belongs to. 
        All classes in the same cluster have the same quantile.
    '''
    # Extract conformal scores for true labels
    cal_class_scores = cal_class_scores[np.arange(len(cal_true_labels)), cal_true_labels]
    
    # Map true class labels to clusters
    cal_true_clusters = np.array([cluster_assignments[label] for label in cal_true_labels])
    
    # Compute cluster qhats
    cluster_qhats = compute_class_specific_qhats(cal_class_scores, cal_true_clusters, alpha=alpha, default_qhat=default_qhat)                           
    # Map cluster qhats back to classes
    num_classes = len(cluster_assignments)
    class_qhats = np.array([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)])
    
    return class_qhats

    # Note: To create prediction sets, just pass class_qhats into create_cb_prediction_sets()
    
# Full Clustered Conformal pipeline for pre-specified n_clustering and num_clusters
def clustered_conformal(totalcal_scores_all, totalcal_labels,
                        alpha,
                        n_clustering, num_clusters,
                        val_scores_all=None, val_labels=None):
    '''
    Use totalcal_scores and total_labels to compute conformal quantiles for each
    class using the clustered conformal procedure. Optionally evaluates 
    performance on val_scores and val_labels
    
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
         
    Outputs:
        - qhats: num_classes-length array where qhats[i] = conformal quantial estimate for class i
        - [Optionally, if val_scores and val_labels are not None] 
            - val_preds: clustered conformal predictions on val_scores
            - val_class_coverage_gap: Class coverage gap, compute on val_scores and val_labels
            - val_set_size_metrics: Dict containing set size metrics, compute on val_scores and val_labels
    '''
    
    num_classes = totalcal_scores_all.shape[1]
    
    # 0) Split data 
    scores1_all, labels1, scores2_all, labels2 = split_X_and_y(totalcal_scores_all, 
                                                               totalcal_labels, 
                                                               n_clustering, 
                                                               num_classes=num_classes, 
                                                               seed=0)
    
    # 1) Compute embedding for each class
    embeddings = embed_all_classes(scores1_all, labels1, q=[0.5, 0.6, 0.7, 0.8, 0.9])
        
    # 2) Cluster classes
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10).fit(embeddings)
    cluster_assignments = kmeans.labels_  
    
    # Print cluster sizes
    print(f'Cluster sizes:', [x[1] for x in Counter(cluster_assignments).most_common()])
    
    # 3) Compute qhats for each cluster
    cal_scores_all = scores2_all
    cal_labels = labels2
    qhats = compute_cluster_specific_qhats(cluster_assignments, 
               cal_scores_all, cal_labels, 
               alpha=alpha, 
               default_qhat=np.inf)
    
    # 4) [Optionally] Apply to val set. Evaluate class coverage gap and set size 
    if (val_scores_all is not None) and (val_labels is not None):
        preds = create_cb_prediction_sets(val_scores_all, qhats)
        class_cov_metrics, set_size_metrics = compute_all_metrics(val_labels, preds, alpha,
                                                                  cluster_assignments=cluster_assignments)
        
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

def automatic_clustered_conformal(totalcal_scores_all, totalcal_labels,
                        alpha,
                        val_scores_all, val_labels,
                        cluster='smart'):
    '''
    Use totalcal_scores_all and total_labels to compute conformal quantiles for each
    class using the clustered conformal procedure. Evaluates 
    performance on val_scores and val_labels
    
    Inputs:
         - totalcal_scores_all: num_instances x num_classes array where 
           cal_class_scores[i,j] = score of class j for instance i
         - totalcal_labels: num_instances-length array of true class labels (0-indexed classes)
         - alpha: number between 0 and 1 that determines coverage level.
         Coverage level will be 1-alpha.
         - val_scores: num_val_instances x num_classes array. Class coverage gap and 
         average set sizes will be computed on val_scores
         and val_labels.
         - val_labels: num_val_instances-length array of true class labels. Class coverage 
         gap and average set sizes will be computed on val_scores 
         and val_labels.
         - cluster: 'smart' or 'always'. In both cases, the clustering settings (n_clustering 
         and num_clusters) are chosen using get_clustering_parameters(). When cluster=='smart',
         we perform a hypothesis test to determine if we should cluster or not 
         cluster using the chosen settings. When cluster=='always' is chosen, we always cluster
         using the chosen settings.
    Outputs:
        - qhats: num_classes-length array where qhats[i] = conformal quantile estimate for class i
        - val_preds: clustered conformal predictions on val_scores
        - val_class_coverage_gap: Class coverage gap, compute on val_scores and val_labels
        - val_set_size_metrics: Dict containing set size metrics, compute on val_scores and val_labels
    '''
    
    num_classes = totalcal_scores_all.shape[1]
    n_totalcal = min(Counter(totalcal_labels).values()) # Count is based on rarest class
    
    # Heuristically choose clustering parameters
    n_clustering, num_clusters = get_clustering_parameters(num_classes, n_totalcal)
    print(f'Heuristic chose n_clustering={n_clustering}, num_clusters={num_clusters}')
    
    
    # If applicable, perform hypothesis test
    if cluster == 'smart':
        pval_threshold = .01
        # Split data between clustering and calibration
        scores1_all, labels1, _, _ = split_X_and_y(totalcal_scores_all, 
                                                               totalcal_labels, 
                                                               n_clustering, 
                                                               num_classes=num_classes, 
                                                               seed=0)
        pval = test_one_cluster_null(scores1_all, labels1, num_classes, num_clusters, 
                                    num_trials=100, seed=0, print_results=False)
        
        if pval < pval_threshold: 
            print(f'p={pval} for one cluster null hypothesis, so running Clustered Conformal')
            # Run clustered conformal and return prediction sets
            qhats, preds, coverage_metrics, set_size_metrics = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                        alpha,
                                                                        n_clustering, num_clusters,
                                                                        val_scores_all=val_scores_all, val_labels=val_labels)
        else:
            print(f'p={pval} for one cluster null hypothesis, so running Standard Conformal')
            # Run Standard Conformal and return prediction sets 
            standard_qhat = compute_qhat(totalcal_scores_all, totalcal_labels, alpha=alpha)
            preds = create_prediction_sets(val_scores_all, standard_qhat)

            coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, preds, alpha)
            qhats = standard_qhat * np.ones((num_classes,))
    elif cluster == 'always':
        # Run clustered conformal and return prediction sets
        qhats, preds, coverage_metrics, set_size_metrics = clustered_conformal(totalcal_scores_all, totalcal_labels,
                                                                    alpha,
                                                                    n_clustering, num_clusters,
                                                                    val_scores_all=val_scores_all, val_labels=val_labels)
    else:
        raise NotImplementedError('Valid options for cluster are "smart" and "always"')
    
    return qhats, preds, coverage_metrics, set_size_metrics
   
            
        
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
    
    # Max gap
    max_gap = np.max(np.abs(class_cond_cov - (1-alpha)))

    # Marginal coverage
    marginal_cov = compute_coverage(val_labels, preds)

    # TODO: Compute average class cov if we leave out classes from smallest cluster
#     # TODO: check if this works in other notebook
#     if cluster_assignments is not None: 
#         cts = Counter(cluster_assignments)
#         smallest_cluster = cts.most_common()[-1][0]
#         classes_not_in_smallest = np.where(cluster_assignments!=smallest_cluster)
#         filtered_class_cond_cov = class_cond_cov[classes_not_in_smallest]
#         filtered_avg_class_cov_gap = np.mean(np.abs(class_cond_cov - (1-alpha)))
        

    class_cov_metrics = {'mean_class_cov_gap': avg_class_cov_gap, 
                         'undercov_gap': undercov_gap, 
                         'overcov_gap': overcov_gap, 
                         'max_gap': max_gap,
                         'marginal_cov': marginal_cov,
                         'raw_class_coverages': class_cond_cov,
                         'cluster_assignments': cluster_assignments # Also save class cluster assignments
                        }

    curr_set_sizes = [len(x) for x in preds]
    set_size_metrics = {'mean': np.mean(curr_set_sizes), '[.25, .5, .75, .9] quantiles': np.quantile(curr_set_sizes, [.25, .5, .75, .9])}
    
    return class_cov_metrics, set_size_metrics