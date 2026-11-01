import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    #first standarise the data 
    n_sample,n_feature=data.shape
    col_mean=np.mean(data,axis=0)
    col_std=np.std(data,axis=0)
    x=(data-col_mean)/col_std
    cov_matrix=np.cov(x,rowvar=False)
    eigenvalues,eigenvectors=np.linalg.eigh(cov_matrix)
    indices=np.flip(np.argsort(eigenvalues))
    egv=eigenvectors[:,indices]
    for j in range(egv.shape[1]):
        for i in range(egv.shape[0]):
            if abs(egv[i][j])>1e-10:
                if egv[i][j]<0:
                    egv[:,j]=-1*egv[:,j]
                    break
    components=np.round(egv[:,:k],decimals=4)
    return components


    pass
