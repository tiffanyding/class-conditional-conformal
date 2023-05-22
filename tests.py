#!/usr/bin/env python
# coding: utf-8

# In[13]:

import warnings
warnings.simplefilter('error')


from utils.clustering_utils import *
from utils.conformal_utils import *
from utils.experiment_utils import *

# Test randomized classwise for achieving exact coverage

# In[4]:


# Load data
softmax_scores, labels = load_dataset('imagenet')
scores_all = 1 - softmax_scores
num_classes = softmax_scores.shape[1]


# In[33]:


alpha = 0.1
avg_num_per_class = 20


# In[34]:


# Split data
seed = 1
scores_all1, labels1, scores_all2, labels2 = random_split(scores_all, labels, avg_num_per_class, seed=seed)


# ## Un-randomized classwise

# In[35]:


classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics = classwise_conformal_pipeline(scores_all1, labels1, 
                                                                                                    scores_all2, labels2, 
                                                                                                    alpha,
                                                                                                    num_classes,
                                                                                                    default_qhat=np.inf, regularize=False)


# In[36]:


coverage_metrics


# ## Randomized classwise

# In[45]:


exact_cov_params, preds, class_cov_metrics, set_size_metrics2 = exact_coverage_classwise_conformal_pipeline(scores_all1, labels1, num_classes, alpha, default_qhat=np.inf, 
                                       val_scores_all=scores_all2, val_labels=labels2)


# In[44]:


np.inf + .2 * np.inf


# In[40]:


class_cov_metrics


# In[43]:


exact_cov_params['q_a'][:10]


# In[41]:


exact_cov_params['q_b'][:10]


# In[42]:


classwise_qhats


# In[47]:


# Compute long-run average raw_class_coverage
num_trials = 10
unrand_class_covs = np.zeros((num_trials, num_classes))
rand_class_covs = np.zeros((num_trials, num_classes))


for i in range(num_trials):
    # Split data
    scores_all1, labels1, scores_all2, labels2 = random_split(scores_all, labels, avg_num_per_class, seed=i)
    
    # Unrandomized classwise
    classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics = classwise_conformal_pipeline(scores_all1, labels1, 
                                                                                                    scores_all2, labels2, 
                                                                                                    alpha,
                                                                                                    num_classes,
                                                                                                    default_qhat=np.inf, regularize=False)
    unrand_class_covs[i,:] = coverage_metrics['raw_class_coverages']
    
    # Randomized classwise
    exact_cov_params, preds, class_cov_metrics, set_size_metrics2 = exact_coverage_classwise_conformal_pipeline(scores_all1, labels1, num_classes, alpha, default_qhat=np.inf, 
                                       val_scores_all=scores_all2, val_labels=labels2)
    rand_class_covs[i,:] = class_cov_metrics['raw_class_coverages']

    
print('UNRAND', np.mean(unrand_class_covs, axis=0))


print('RAND', np.mean(rand_class_covs, axis=0))

print('UNRAND', np.mean(unrand_class_covs, axis=1))


print('RAND', np.mean(rand_class_covs, axis=1))

