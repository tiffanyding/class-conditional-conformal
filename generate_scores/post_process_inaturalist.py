# NOT RUNNABLE. Code copied from load dataset. 

# load iNaturalist softmax scores, then

thresh = 250 # changed from 150
softmax_scores, labels = remove_rare_classes(softmax_scores, labels, thresh=250)

print('New softmax_scores shape:', softmax_scores.shape) 
    

# TODO: save

'''
def load_dataset(dataset, remove_rare_cls=False):

    Inputs:
        - dataset: string specifying dataset. Options are ['imagenet', 'cifar-100', 'places365', 'inaturalist']

    if dataset == 'imagenet':
        softmax_path = '/home/tding/data/finetuned_imagenet/imagenet_train_subset_softmax.npy'
        labels_path = '/home/tding/data/finetuned_imagenet/imagenet_train_subset_labels.npy'
    elif dataset == 'cifar-100':
        softmax_path = "/home/tding/code/class-conditional-conformal-datasets/notebooks/.cache/best-cifar100-model-fracval=0.5-valsoftmax_frac=0.5.npy"
        labels_path = "/home/tding/code/class-conditional-conformal-datasets/notebooks/.cache/best-cifar100-model-fracval=0.5-vallabels_frac=0.5.npy"
#         softmax_path = "/home/tding/code/class-conditional-conformal-datasets/notebooks/.cache/best-cifar100-model-fracval=0.7-valsoftmax_frac=0.7.npy"
#         labels_path = "/home/tding/code/class-conditional-conformal-datasets/notebooks/.cache/best-cifar100-model-fracval=0.7-vallabels_frac=0.7.npy"
    elif dataset == 'places365':
        softmax_path = '/home/tding/code/class-conditional-conformal-datasets/notebooks/.cache/best-Places365-model-valsoftmax_frac=0.1.npy'
        labels_path = '/home/tding/code/class-conditional-conformal-datasets/notebooks/.cache/best-Places365-model-vallabels_frac=0.1.npy'
    elif dataset == 'inaturalist':
        # 'family' level
        softmax_path = '/home/tding/code/class-conditional-conformal-datasets/notebooks/.cache/best-iNaturalist-model-valsoftmax_frac=0.5.npy'
        labels_path = '/home/tding/code/class-conditional-conformal-datasets/notebooks/.cache/best-iNaturalist-model-vallabels_frac=0.5.npy'
    
#         # full species level (6414 classes before filtering)
#         softmax_path = '../class-conditional-conformal-datasets/notebooks/.cache/archived/best-iNaturalist-model-valsoftmax_frac=0.5.npy'
#         labels_path = '../class-conditional-conformal-datasets/notebooks/.cache/archived/best-iNaturalist-model-vallabels_frac=0.5.npy'
    
        
        
        remove_rare_cls = True

    softmax_scores = np.load(softmax_path)
    labels = np.load(labels_path)
    
    print('softmax_scores shape:', softmax_scores.shape) 
    
    if remove_rare_cls:
        thresh = 250 # changed from 150
        softmax_scores, labels = remove_rare_classes(softmax_scores, labels, thresh=250)
        
        print('New softmax_scores shape:', softmax_scores.shape) 
    
    return softmax_scores, labels
'''
