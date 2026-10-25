import numpy as np
from typing import Tuple
def gini_impurity(x):
    freq={}
    n=len(x)
    for elm in x:
        freq[elm]=freq.get(elm,0)+1
    gini=1
    for u,v in freq.items():
        prob=v/n
        gini-=prob**2
    return gini

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    n,d =X.shape
    best_feature=0
    best_threshold=np.min(X[:,0])
    best_gini=float("inf")
    for j in range(d):
        #removing duplicate values and sorted in ascending order
        thresholds=np.unique(X[:,j])

        for i,threshold in enumerate(thresholds):
            left_y=y[X[:,j]<=threshold]
            right_y=y[ X[:,j]> threshold]
            lft_gini=gini_impurity(left_y)
            rt_gini=gini_impurity(right_y)
            weighted_gini=(len(left_y)*lft_gini + len(right_y) *rt_gini)/n
            if weighted_gini<best_gini:
                best_gini=weighted_gini
                best_feature=j
                best_threshold=threshold
    return (best_feature,best_threshold)
            

    pass