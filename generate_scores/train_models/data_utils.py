import numpy as np

def split_X_and_y(X, y, n_k, num_classes, seed=0):
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