import numpy as np
from collections import defaultdict
def stratified_kfold_indices(y, n_splits):
	"""
	Generate train/test indices for stratified K-fold cross-validation.

	Args:
		y: 1D array-like of integer class labels
		n_splits: number of folds

	Returns:
		A list of [train_indices, test_indices] pairs, one per fold.
	"""
	classes,counts=np.unique(y,return_counts=True)
	if np.min(counts)< n_splits:
		raise ValueError("Invalid n_splits ")
	#now i need to collect each class indices
	classes_indices=defaultdict(list)
	for i,cls in enumerate(y):
		classes_indices[cls].append(i)
	
	#now for each class i need to split them and add corresponding value 
	groups=[]
	for cls in classes:
		cls_indices=classes_indices[cls]
		q=len(cls_indices)//n_splits
		r=len(cls_indices)%n_splits
		start=0
		class_group=[]
		for i in range(n_splits):
			end=start+q
			if i<r:
				end+=1
			class_group.append(cls_indices[start:end])
			start=end
		groups.append(class_group)
	
	splits=[]
	for i in range(n_splits):
		test_indices=[]

		for j in range(len(classes)):
			test_indices.extend(groups[j][i])
		
		train_indices=list(sorted(set(range(len(y)))-set(test_indices)))

		splits.append([train_indices,sorted(test_indices)])
	return splits
		
	
	pass