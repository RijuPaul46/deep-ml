import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Your code here
    X=np.array(X)
    n_feature,n_sample=X.shape
    cov_mat=np.cov(X,rowvar=False)
    eigenval,eigenvectors=np.linalg.eigh(cov_mat)
    total_eig=np.sum(eigenval)
    eigenvalue=np.flip(np.sort(eigenval))
    ratio=eigenvalue/total_eig
    return ratio
    pass