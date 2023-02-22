import joblib 
import numpy as np

from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF # Empirical CDF

#========================================
#   Misc.
#========================================

# No longer used
def transform_to_standard_normal(samples):
    F = ECDF(np.concatenate((samples.flatten(), [-np.inf, np.inf])), side='right')
    F_samples = F(samples)
    final_samples = stats.norm.ppf(F_samples)

    return final_samples

#========================================
#   Computing distances
#========================================

def compute_distance_metric(samples1, samples2, dist_metric, 
                            quantiles=None, weights=None):
    '''
    Computes some notion of distance between two samples 
    
    Inputs:
        - samples1, samples2: Two arrays of samples 
        - dist_metric: Distance metric to use. Options include
            * 'KS_pvalue': 1 - (Kolmogorov-Smirnov p-value)
            * 'KS_statistic': Kolmogorov-Smirnov test-statistic
            * 't_test_statistic': Absolute value of test statistic for two-sample t-test
            * 'quantile_statistic': Let Q_1(q) and Q_2(q) denote the q-th quantile of 
            samples1 and samples2 respectively. The statistic is computed as 
            QS = \sum_i (weights[i] * |Q_1(quantiles[i]) - Q_2(quantiles[i])|)
        - quantiles: When dist_metric = 'quantile_statistic', this is the list of 
          quantiles at which the quantile gap will be computed. Otherwise, this
          argument is ignored
        - weights: When dist_metric = 'quantile_statistic', this is the list of 
          weights to be used to compute the statistic. weights[i] is the weight
          of the i-th quantile
    '''
    if dist_metric == 'KS_pvalue':
        results = stats.kstest(samples1, samples2, alternative='two-sided')
        dist = 1 - results.pvalue
    elif dist_metric == 'KS_statistic':
        results = stats.kstest(samples1, samples2, alternative='two-sided')
        dist = results.statistic
    elif dist_metric == 't_test_statistic': 
        results = stats.ttest_ind(samples1, samples2)
        dist = np.abs(results.statistic)
    elif dist_metric == 'quantile_statistic':
        if quantiles == None: 
            quantiles = [0.5, 0.6, 0.7, 0.8, 0.9]
        if weights == None:
            # Default: uniform weights
            weights = np.ones((len(quantiles),)) / len(quantiles)
            
        dist = 0
        for i in range(len(quantiles)):
            q = quantiles[i]
            Q1 = np.quantile(samples1, q) 
            Q2 = np.quantile(samples2, q) 
            dist += weights[i] + np.abs(Q1 - Q2)
        
        # Normalization to account for sample size
        n1 = len(samples1)
        n2 = len(samples2)
        dist *= ((1/n1 + 1/n2) ** (-1/2))
            
    return dist

# No longer used
def kolmogorov_smirnov_pvalue(i, j, scores, labels):
    class_i_scores = scores[labels==i,i]
    class_j_scores = scores[labels==j,j]
    
    # result stores the Kolmogorov test statistic and the p-value
    result = stats.kstest(class_i_scores, class_j_scores, alternative='two-sided')
    
    return result.pvalue

def compute_distance_matrix(scores1_all, labels1, num_classes, n_jobs=4):
    pvalues = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(kolmogorov_smirnov_pvalue)(i, j, scores1_all, labels1) for j in range(num_classes) for i in range(num_classes))
    distances = 1 - np.array(pvalues)
    distances = np.reshape(distances, (num_classes, num_classes))
    
    return distances


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