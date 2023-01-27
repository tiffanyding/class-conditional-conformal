# Copied from empirical-bayes-conformal repo and modified

import matplotlib.pyplot as plt
import numpy as np
import torch

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

#     np.random.seed(seed)
    
#     X1 = np.zeros((num_classes * n_k, num_classes))
#     y1 = np.zeros((num_classes * n_k, ), dtype=np.int32)
    
#     all_selected_indices = np.zeros(y.shape)

#     for k in range(num_classes):

#         # Randomly select n instances of class k
#         idx = np.argwhere(y==k).flatten()
#         selected_idx = np.random.choice(idx, replace=False, size=(n_k,))

#         X1[n_k*k:n_k*(k+1), :] = X[selected_idx, :]
#         y1[n_k*k:n_k*(k+1)] = k
        
#         all_selected_indices[selected_idx] = 1
        
#     X2 = X[all_selected_indices == 0]
#     y2 = y[all_selected_indices == 0]
    
#     return X1, y1, X2, y2

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
#   Clustered conformal inference
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