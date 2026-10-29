import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing=smoothing
        pass

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        #first calculate prior probablity for each class
        n_sample,n_feature=X.shape
        class_count=np.bincount(y.astype(int),minlength=2)
        class_prob=class_count/n_sample
        present=class_prob>0
        self.log_prior=np.full(2,-np.inf)
        self.log_prior[present]=np.log(class_prob[present])
        #now we need to calculate for each class 
        self.feature_prob=np.zeros((2,n_feature))
        for c in range(2):
            X_c=X[y==c]
            feature_count=np.sum(X_c,axis=0)
            #due to smoothing feature_prob will never be equal to 0
            self.feature_prob[c]=(
                (feature_count+self.smoothing)/(len(X_c)+2*self.smoothing)
            )
            
        self.log_feature_present=np.log(self.feature_prob)
        self.log_feature_absent=np.log(1-self.feature_prob)
        pass

    def predict(self, X):
        # Predict class labels for test set X
        predictions=[]
        for x in X:
            class_prob=self.log_prior
            for c in range(2):
                for j,val in enumerate(x):
                    if val==1:
                        class_prob[c]+=self.log_feature_present[c][j]
                    else:
                        class_prob[c]+=self.log_feature_absent[c][j]
            predictions.append(np.argmax(class_prob))
        return np.array(predictions)
        pass