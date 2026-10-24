import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    train_errors=[]
    val_errors=[]

    for n in train_sizes:
        X_feature=np.array(X_train[:n])[:,0]
        y=np.array(y_train[:n])
        X=np.vander(X_feature,N=degree+1,increasing=True)
        w=np.linalg.pinv(X) @ y
        train_pred= X @ w 
        train_mse=(1/n)*np.sum((train_pred-y)**2)
        val_x=np.vander(np.array(X_val)[:,0], N=degree+1,increasing=True)
        val_y=np.array(y_val)
        val_pred= val_x @ w
        val_mse=np.mean((val_pred-val_y)**2)
        train_errors.append(train_mse)
        val_errors.append(val_mse)
    last_train=train_errors[-1]
    last_val=val_errors[-1]
    diagnosis=""
    if last_train>bias_threshold:
        diagnosis="high_bias"
    elif last_val - last_train > variance_threshold:
        diagnosis="high_variance"
    else :
        diagnosis="good_fit"
    return {
        "train_errors":train_errors,
        "val_errors": val_errors,
        "diagnosis": diagnosis
    }


    pass