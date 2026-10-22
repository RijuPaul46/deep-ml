import numpy as np
from itertools import combinations_with_replacement
from math import prod
def polynomial_features(X, degree):
    # ✏️  Your code here
    n_sample,n_feature=X.shape
    result=[]
    for row in X:
        features=[1.0]
        for d in range(1,degree+1):
            for combo in combinations_with_replacement(range(n_feature),d):
                value=prod(row[i] for i in combo)
                features.append(value)
        features.sort()
        result.append(features)
    return np.array(result)
    pass