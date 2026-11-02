import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    indices=np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    folds=np.full(k,n_samples//k)
    rem=n_samples%k
    for i in range(k):
        if rem==0:
            break
        folds[i]+=1
        rem-=1
    validation=[]
    start=0
    for i,no in enumerate(folds):
        test_indices=indices[start:start+no]
        train_indices=np.concatenate((
            indices[0:start],
            indices[start+no:]
        ))
        start+=no
        validation.append((train_indices,test_indices))
    return validation
    pass