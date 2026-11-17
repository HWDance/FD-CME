from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier,GradientBoostingClassifier
import numpy as np
from utils import Scalar, Array

class table_propensity:
    def __init__(self, xdim:Scalar):
        #initialise the probability table
        self.xdim=xdim
        if xdim == 1:
            self.opts = np.array([[0],
                                  [1]])
        elif xdim == 2:
            self.opts = np.array([[0,0],
                                  [0,1],
                                  [1,0],
                                  [1,1]])
        self.table=np.zeros((2**xdim,2))
        
    def fit(self,X:Array,A:Array):
        #fit the propensities by updating the table
        for i in range(2**self.xdim):
            x_val=self.opts[i,:]
            indices=np.where(np.all(X==x_val,axis=1))
            A_x=A[indices]
            ones=A_x.sum()
            zeros=A_x.size-ones
            self.table[i,0]=zeros
            self.table[i,1]=ones
        return self.table

    def predict_proba(self,X:Array):
        #predict probabilities using the table
        #output should give probability of A=0 and probability of A=1 for given X
        n=X.shape[0]
        output_table=np.zeros((n,2))
        for i in range(2**self.xdim):
            x_val=self.opts[i,:]
            indices=np.where(np.all(X==x_val,axis=1))
            probs = self.table[i,:]/self.table[i,:].sum()
            output_table[indices]=probs
        return output_table

def estimate_propensity(A:Array, X:Array, method:str='logistic'):
    #return a callable function f such that f(a,x)=p(A=a|X=x)
    n_features = X.shape[1]
    n_steps=100

    if method=='table':
        f=table_propensity(n_features)
    
    elif method=='logistic':
        f=LogisticRegressionCV(
            cv=5, #5-fold cross-validation
            random_state=0,
            l1_ratios=(0,),
            scoring='neg_log_loss',
            use_legacy_attributes=False,
            max_iter=n_steps
        )

    elif method=='forest':
        f = GridSearchCV(
            RandomForestClassifier(
                n_estimators=1000,
                criterion='log_loss',
                n_jobs=8
            ),
            param_grid={"max_depth": [1,3,5,10]},
            scoring='neg_log_loss'
        )

    elif method=='histgbm':
        f = GridSearchCV(
            HistGradientBoostingClassifier(
                max_iter=1000,
                max_leaf_nodes=None,
                min_samples_leaf=1,
                scoring='neg_log_loss',
                validation_fraction=0.2,
                early_stopping=True
            ),
            param_grid={"max_depth": [3,5,10],"learning_rate": [0.01,0.1,0.5]},
            scoring='neg_log_loss',
            n_jobs=8
        )

    elif method=='reggbm':
        f = GridSearchCV(
            GradientBoostingClassifier(
                n_estimators=1000,
                max_leaf_nodes=None,
                min_samples_leaf=1,
                validation_fraction=0.2,
                n_iter_no_change=10
            ),
            param_grid={"max_depth": [3,5,10],"learning_rate": [0.01,0.1,0.5]},
            scoring='neg_log_loss',
            n_jobs=8
        )
        
    f.fit(X,np.ravel(A))
        
    return f #can predict the log probabilities of each a in A for a given X=x by calling f.predict_log_proba(x)