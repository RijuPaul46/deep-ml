import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	# Your code here
	n_sample,n_feature=X.shape

	#no of class 
	C=np.max(y)+1

	#one hot encoded 
	Y=np.eye(C)[y]
	#add one col for bias term 
	x=np.column_stack((np.ones(n_sample),X))

	# define weight matrix
	W=np.zeros((C,n_feature+1))
	
	losses=[]
	for _ in range(iterations):

		Z=x @ W.T
		Z_shifted=Z-np.max(Z,axis=1,keepdims=True)
		exp_z=np.exp(Z_shifted)
		P=exp_z/np.sum(exp_z,axis=1,keepdims=True)
		corrected_prob=P[np.arange(n_sample),y]
		loss=np.sum(-np.log(corrected_prob))
		losses.append(loss)
		gradient=(P-Y).T @ x
		W=W-learning_rate*gradient
		
		
	return W.tolist(),losses
	pass