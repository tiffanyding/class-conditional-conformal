# This only needs to be run once, to generate the files for the Google Drive upload

import sys; sys.path.append("../") # For relative imports

from utils.experiment_utils import *

save_folder = '/checkpoints/tding/data/data'

# for dataset in ['imagenet','cifar-100', 'places365', 'inaturalist']:
for dataset in ['inaturalist']:
    softmax_scores, labels = load_dataset(dataset)
    np.savez(f'{save_folder}/{dataset}.npz', softmax=softmax_scores, labels=labels)
    print('Saved to', f'{save_folder}/{dataset}.npz')