import glob # For getting file names
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
# import seaborn as sns
# import torch

from collections import Counter
from scipy import stats, cluster
from sklearn.cluster import KMeans

from utils.clustering_utils import *
from utils.conformal_utils import *
from utils.experiment_utils import *

# iNaturalist
softmax_path = '../class-conditional-conformal-datasets/notebooks/.cache/best-iNaturalist-model-valsoftmax_frac=0.5.npy'
labels_path = '../class-conditional-conformal-datasets/notebooks/.cache/best-iNaturalist-model-vallabels_frac=0.5.npy'
save_folder = '.cache/paper/places365'

# # ImageNet
# softmax_path = '/home/tding/data/finetuned_imagenet/imagenet_train_subset_softmax.npy'
# labels_path = '/home/tding/data/finetuned_imagenet/imagenet_train_subset_labels.npy'
# save_folder = '.cache/paper/imagenet'

# # CIFAR-100
# softmax_path = "../class-conditional-conformal-datasets/notebooks/.cache/best-cifar100-model-valsoftmax_frac=0.3.npy"
# labels_path = "../class-conditional-conformal-datasets/notebooks/.cache/best-cifar100-model-vallabels_frac=0.3.npy"
# save_folder = '.cache/paper/cifar100'
 
# SETTINGS
alpha = .1
n_totalcal_list = [10, 30]
score_function_list = ['softmax', 'APS']
seeds = [0,1,2,3,4] # 5,6,7,8,9]

softmax_scores = np.load(softmax_path)
labels = np.load(labels_path)

print('softmax_scores shape:', softmax_scores.shape) # (993763, 6414)
print('Class counts:', Counter(labels).most_common())

run_experiment(softmax_scores, labels,
                  save_folder,
                  alpha=alpha,
                  n_totalcal_list=n_totalcal_list,
                  score_function_list = score_function_list,
                  seeds=seeds)

for n_totalcal in n_totalcal_list:
    for score in score_function_list:
        print(f'===== n_totalcal={n_totalcal}, score={score} =====')
        folder = f'{save_folder}/n_totalcal={n_totalcal}/score={score}/'
        df = average_results_across_seeds(folder)
        print(df)

