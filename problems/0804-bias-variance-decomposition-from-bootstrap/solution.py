import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    pred=np.array(predictions)
    y_true=np.array(y_true)
    B,M=pred.shape
    bias_sqrd=0.0
    variance=0.0
    total_mse=0.0
    for j in range(M):
        col=pred[:,j]
        col_mean=np.mean(col)
        sq_bias=(col_mean-y_true[j])**2
        var=np.mean((col-col_mean)**2)
        mse=np.mean((col-y_true[j])**2)
        bias_sqrd+=sq_bias
        variance+=var
        total_mse+=mse
    bias_squared=bias_sqrd/M 
    variance/=M 
    mse=total_mse/M 
    return {
        "bias_squared":bias_squared,
        "variance":variance,
        "mse":mse
    }
    pass
