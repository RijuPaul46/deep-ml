import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    n_sample,n_feature=X.shape
    col_mean=np.mean(X,axis=0)
    X_c=X-col_mean
    cov_mat= (X_c.T @ X_c)/n_sample
    eigenvalues,eigenvectors=np.linalg.eigh(cov_mat)
    indices=np.argsort(eigenvalues)[:: -1]
    egv=eigenvectors[:,indices]
    components=egv[:,:n_components]
    projected=X_c @ components 
    constructed=projected @ components.T + col_mean
    mse=0
    for i in range(n_sample):
        for j in range(n_feature):
            mse+=(X[i][j]-constructed[i][j])**2
    mse/=(n_feature*n_sample)
    return mse
    pass