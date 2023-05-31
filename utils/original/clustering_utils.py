import joblib 
import numpy as np

from scipy import stats
from sklearn.cluster import KMeans

#========================================
#   Misc.
#========================================

def get_true_class_conformal_score(scores_all, labels):
    '''
    Extracts conformal scores that corresponds to the true class labels
    
    Inputs:
        scores_all: n x num_classes array 
        labels: length-n array of true class labels
    '''
    return scores_all[np.arange(len(labels)), labels]

#========================================
#  Testing null hypothesis of one cluster
#========================================

# from sklearn import metrics 


# def compute_avg_distance_between_quantiles(list_of_arrs, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
#     '''
#     Computes the L1 distance between quantiles q between each pair of groups
#     in list_of_arrs and then takes average across all pairs
    
#     Input:
#         list_of_arrs: length-n list of arrays. list_of_arrs[i] contains 
#         samples from group i
#     '''
#     n_groups = len(list_of_arrs)
    
#     dists = []
#     for i in range(n_groups):
        
#         groupi_quantiles = np.quantile(list_of_arrs[i], q)
        
#         for j in range(i+1, n_groups):
#             groupj_quantiles = np.quantile(list_of_arrs[j], q)
            
#             dist_ij = np.sum(np.abs(groupi_quantiles - groupj_quantiles))
#             dists.append(dist_ij)
            
#     avg_dist = sum(dists) / len(dists)
#     return avg_dist

# def _get_cluster_fit(true_class_scores, labels, num_classes, num_clusters):
#     '''
#     Cluster scores = union of scores for each class in the cluster
#     Compute quantile embedding on cluster scores
#     Compute average pairwise distance between cluster embeddings
    
#     Under null hypothesis (i.e., no clusters), the pairwise distances should 
#     be approximately 0. However, if there are clusters, the distances should 
#     be > 0. 
#     '''
    
#     # Compute embeddings
#     q = [0.5, 0.6, 0.7, 0.8, 0.9]
#     embeddings = np.zeros((num_classes, len(q)))
#     for i in range(num_classes):
#         class_i_scores = true_class_scores[labels==i]
#         embeddings[i,:] = quantile_embedding(class_i_scores, q=q)

#     kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(embeddings)
#     cluster_labels = kmeans.labels_
    
#     # Average L1 distance of quantiles of original scores in each cluster ???
#     list_of_cluster_scores = []
#     for i in range(num_clusters):
#         clusteri_classes = np.argwhere(cluster_labels == i)
#         clusteri_scores = true_class_scores[np.in1d(labels, clusteri_classes)]
#         list_of_cluster_scores.append(clusteri_scores)
#     cluster_fit_metric = compute_avg_distance_between_quantiles(list_of_cluster_scores, q=q)
    
#     return cluster_fit_metric


# def test_one_cluster_null(scores, labels, num_classes, num_clusters, 
#                           num_trials=100, seed=0, print_results=False):
#     np.random.seed(seed)
    
#     if len(scores.shape) > 1:
#         true_class_scores = get_true_class_conformal_score(scores, labels)
#     else:
#         true_class_scores = scores
    
#     # Compute metric using true class labels
#     observed_metric = _get_cluster_fit(true_class_scores, labels, num_classes, num_clusters)   
    
#     metrics_under_null = np.zeros((num_trials,))
#     permuted_labels = np.copy(labels)
#     for i in range(num_trials):
#         # Randomly permute labels
#         np.random.shuffle(permuted_labels)
        
#         # Compute metric for each random permutation 
#         metrics_under_null[i] = _get_cluster_fit(true_class_scores, permuted_labels, num_classes, num_clusters)
        
#     # Compute fraction of results under null that yield a better clustering metric 
#     # than the observed value 
#     num_better = np.sum(metrics_under_null > observed_metric) # Lower inertia = better clustering
#     p_value = num_better / num_trials
    
#     if print_results:
#         print('Observed metric:', observed_metric)
#         print('Metric under null:', metrics_under_null)

#         print(f'\nProbability of observing a larger metric under null hypothesis of one cluster: {p_value}',
#               f'({num_better} out of {num_trials} trials)')
    
#     return p_value

#========================================
#   Computing embeddings for k-means
#========================================

def quantile_embedding(samples, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
    '''
    Computes the q-quantiles of samples and returns the vector of quantiles
    '''
    return np.quantile(samples, q)

def embed_all_classes(scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
    '''
    Input:
        - scores_all: num_instances x num_classes array where 
            scores_all[i,j] = score of class j for instance i
          Alternatively, num_instances-length array where scores_all[i] = score of true class for instance i
        - labels: num_instances-length array of true class labels
        - q: quantiles to include in embedding
        - return_cts: if True, return an array containing the counts for each class 
        
    Output: 
        - embeddings: num_classes x len(q) array where ith row is the embeddings of class i
        - (Optional) cts: num_classes-length array where cts[i] = # of times class i 
        appears in labels 
    '''
    num_classes = len(np.unique(labels))
    
    embeddings = np.zeros((num_classes, len(q)))
    cts = np.zeros((num_classes,))
    
    for i in range(num_classes):
        if len(scores_all.shape) == 2:
            class_i_scores = scores_all[labels==i,i]
        else:
            class_i_scores = scores_all[labels==i] 
        cts[i] = class_i_scores.shape[0]
        embeddings[i,:] = quantile_embedding(class_i_scores, q=q)
    
    if return_cts:
        return embeddings, cts
    else:
        return embeddings


#========================================
#   Generating synthetic data
#========================================

def generate_synthetic_clustered_data(num_clusters, num_classes, num_samples_per_class, 
                                      cluster_probs=None, dist_between_means=1000, sd=1):
    '''
    Generate clusters where cluster i is a N(i*dist_between_means, 1) distribution 
    Randomly assign classes to clusters with probabilities determined by cluster_probs. Then sample 
    num_samples_per_class from each class.
    
    Inputs:
        - num_clusters: Number of clusters
        - num_classes: Total number of classes
        - num_samples_per_class: Number of samples to generate per class
        - cluster_probs: If None, then every class has equal probability of being assigned 
            to each cluster. Otherwise, it must be an array of probabilities of length num_clusters
            such that cluster_probs[i] = probability that a class is assigned to cluster i
        - dist_between_means: Distance between means of Normal distributions
        - sd = Standard deviation of Normal distributions
            
    Output: cluster_assignments, samples
        - cluster_assignments: (num_classes,) array of cluster assignments 
        - samples: (num_classes, num_samples_per_class) array containing the generated samples
    '''
    cluster_assignments = np.zeros((num_classes,))
    samples = np.zeros((num_classes, num_samples_per_class))
    
    for i in range(num_classes):
        cluster_assignments[i] = np.random.choice(np.arange(num_clusters), p=cluster_probs)
        samples[i,:] = np.random.normal(loc=cluster_assignments[i] * dist_between_means, 
                                        scale=sd,
                                        size=(num_samples_per_class,))
        
    return cluster_assignments, samples


def sample_from_empirical_distr(data, num_samples):
    samples = np.random.choice(data, size=num_samples)
    
    return samples

def generate_realistic_clustered_data(samples_list, 
                                      num_classes, 
                                      num_samples_per_class, 
                                      cluster_probs=None):
    '''
    Generate clusters where cluster i has the same distribution as the samples in samples_list[i]. 
    Randomly assign classes to clusters with probabilities determined by cluster_probs. Then sample 
    num_samples_per_class from each class.
    
    Inputs:
        - samples_list: num_cluster length list, where samples_list[i] is an
          array of samples from distribution i
        - num_classes: Total number of classes
        - num_samples_per_class: Number of samples to generate per class
        - cluster_probs: If None, then every class has equal probability of being assigned 
            to each cluster. Otherwise, it must be an array of probabilities of length num_clusters
            such that cluster_probs[i] = probability that a class is assigned to cluster i
            
    Output: cluster_assignments, samples
        - cluster_assignments: (num_classes,) array of cluster assignments 
        - samples: (num_classes, num_samples_per_class) array containing the generated samples
    '''
    
    num_clusters = len(samples_list)
    
    cluster_assignments = np.zeros((num_classes,), dtype=int)
    samples = np.zeros((num_classes, num_samples_per_class))
    for i in range(num_classes):
        cluster_assignments[i] = np.random.choice(np.arange(num_clusters), p=cluster_probs)
        samples[i,:] = sample_from_empirical_distr(samples_list[cluster_assignments[i]], num_samples_per_class)
        
    return cluster_assignments, samples