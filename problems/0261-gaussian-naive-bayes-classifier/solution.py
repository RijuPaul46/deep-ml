import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	# Your code here
	n_sample,n_feature=X_train.shape
	#return each unique val in y_train and their count in counts
	classes,counts=np.unique(y_train,return_counts=True)

	log_prior=np.log(counts/len(y_train))

	means=[]
	variance=[]

	for c in classes:
		X_c=X_train[y_train==c]

		mean=np.mean(X_c,axis=0)

		varnc=np.var(X_c,axis=0)

		means.append(mean)
		variance.append(varnc)
	# not added for which class ... because we will iterate in same manner so indexing not needed for this 
	means=np.array(means)

	variance=np.array(variance)

	predictions=[]

	for x in X_test:

		scores=[]

		for i,c in enumerate(classes):
			var=variance[i]
			mean=means[i]
			score=log_prior[i]
			score+=np.sum(-0.5*(
				np.log(np.pi*2*var)+(x-mean)**2/var
			))
			scores.append(score)

		predictions.append(int(classes[np.argmax(scores)]))
	return predictions


	pass