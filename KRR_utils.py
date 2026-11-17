import jax
import optax
import jax.numpy as jnp
from jax import vmap,grad,random
from flax import struct
import numpy as np
import abc
from kernels import gaussian_kernel,_l2_norm_squared
from utils import Array, Scalar, base_kernel

def ard_median_heuristic(X:Array):
    """per-dimension median heuristic, scaled so that it matches the isotropic median heuristic"""
    def differences(x,y):
        return jnp.square(x-y)

    D2 = vmap(vmap(differences, (None, 0)), (0, None))(X, X)

    ards_ls = vmap(med_from_array, in_axes=-1)(D2)

    iso_ls = iso_median_heuristic(X)+1e-12
    ards_norm = jnp.linalg.norm(ards_ls)+1e-12

    lengthscales = (ards_ls+1e-12*jnp.ones(ards_ls.shape)) * iso_ls / ards_norm
    return lengthscales
    

def iso_median_heuristic(X:Array):
    """isotropic median heuristic (i.e., median of overall pairwise distances, not per-dimension)"""
    def differences(x,y):
        return jnp.sum(jnp.square(x-y))

    D2 = vmap(vmap(differences, (None, 0)), (0, None))(X, X)

    return med_from_array(D2)

def med_from_array(D2:Array):
    """helper function"""
    mask=jnp.tril(jnp.ones_like(D2,dtype=jnp.bool),k=-1)
    median = jnp.median(D2[mask])
    return jnp.sqrt(median/2.)

class KRR:
    def __init__(self,Y_kernel:base_kernel,jitter:Scalar=1e-6):
        self.Y_kernel = Y_kernel
        self.jitter=jitter

    def __call__(self, x: Array, y: Array, theta: Array) -> Array:
        return jnp.exp(- 0.5*_l2_norm_squared(jnp.divide((x - y),theta)))

    def make_distance_matrix(self, X: Array, Y: Array, theta:Array) -> Array:
        #modified from base_kernel class to include theta
        return vmap(vmap(type(self).__call__, (None, None, 0, None)), (None, 0, None, None))(self, X, Y, theta)

    def linear_weights(self,X_tr:Array, X_te:Array, theta:Array):
        K_Xx=self.make_distance_matrix(X_tr,X_te,theta[1:])
        n = X_tr.shape[0]
        K_XX_reg = self.make_distance_matrix(X_tr,X_tr,theta[1:])+(theta[0]+self.jitter)*jnp.eye(n)
        L=jax.scipy.linalg.cho_factor(K_XX_reg)
        return jax.scipy.linalg.cho_solve(L,K_Xx)
        
        
    def loss(self,X_tr:Array,Y_tr:Array,X_te:Array,Y_te:Array,theta:Array):
        K_Xx=self.make_distance_matrix(X_tr,X_te,theta[1:])
        n = X_tr.shape[0]
        K_XX_reg = self.make_distance_matrix(X_tr,X_tr,theta[1:])+(theta[0]+self.jitter)*jnp.eye(n)
        L=jax.scipy.linalg.cho_factor(K_XX_reg)
        #S = self.linear_weights(X_tr,X_te,theta)
        S=jax.scipy.linalg.cho_solve(L,K_Xx)
        m=X_te.shape[0]

        K_tr_tr = self.Y_kernel.make_distance_matrix(Y_tr,Y_tr)
        K_tr_te = self.Y_kernel.make_distance_matrix(Y_tr,Y_te)
        K_te_te = self.Y_kernel.make_distance_matrix(Y_te,Y_te)

        term_1 = jnp.diag(K_te_te).sum()
        term_2 = jnp.trace(K_tr_te.T@S)
        term_3 = jnp.trace(S.T@K_tr_tr@S)

        return (term_1-2.*term_2+term_3)/m

def subsample_train_fold(X:Array,
                         Y:Array,
                         subsamples:int=1000,
                         rng_key:Array=None):
    """
    Subsample ONLY from the training fold (never from validation).
    """
    n = X.shape[0]
    if subsamples is None or subsamples >= n:
        return X, Y
    if rng_key is None:
        seed = 0
        rng_key = jax.random.PRNGKey(seed)
            
    perm = jax.random.permutation(rng_key,n)[:subsamples]
    return X[perm], Y[perm]

def get_CV_splits(X:Array,
                  Y:Array,
                  nfolds:int=3,
                  rng_key:Array=None):
    """
    Returns list of (train_data, val_data) disjoint folds covering {0,...,n-1}.
    """
    n = X.shape[0]
    nfolds = int(max(1, min(nfolds, n)))

    if rng_key is None:
        seed = 0
        rng_key = jax.random.PRNGKey(seed)
        
    perm = jax.random.permutation(rng_key,n)

    # equal partitions so we can use vmap on resulting datasets
    fold_size = n // nfolds
    folds = []
    start = 0
    for _ in range(nfolds):
        end = start + fold_size
        folds.append(perm[start:end])
        start = end

    splits = []
    for k in range(nfolds):
        val_idx = folds[k]
        train_idx = jnp.concatenate(
            [folds[j] for j in range(nfolds) if j != k], axis=0
        )
        splits.append((train_idx, val_idx))

    X_tr_list=[]
    Y_tr_list=[]
    X_val_list=[]
    Y_val_list=[]
    for train_idx, val_idx in splits:
        X_tr_list.append(X[train_idx])
        Y_tr_list.append(Y[train_idx])
        X_val_list.append(X[val_idx])
        Y_val_list.append(Y[val_idx])

    X_tr = jnp.dstack(X_tr_list)
    Y_tr = jnp.dstack(Y_tr_list)
    X_val = jnp.dstack(X_val_list)
    Y_val = jnp.dstack(Y_val_list)
    return X_tr,Y_tr,X_val,Y_val


def optimise(learner,
             X:Array,
             Y:Array,
             step_num:int = 100,
             nfolds:int = 3,
             step_size:float = 1e-2, 
             subsampling:bool=False,
             subsamples:int=1000,
             rng_key:Array=None):
    """
        Gradient-based tuning of functional hyperparameters by K-fold
        kernel-RKHS CV risk.

        - For each epoch, we iterate K folds:
            optionally subsample the training fold,
            fit the functional on the training fold,
            compute RKHS loss on the validation fold using S = linear_weights,
            accumulate mean CV loss,
            backprop once per epoch.
    """
    params = jnp.ones((X.shape[1]+1,))

    optimizer = optax.adam(learning_rate=step_size)
    opt_state = optimizer.init(params)

    if subsampling:
        X,Y=subsample_train_fold(X,Y,subsamples,rng_key)

    X_tr,Y_tr,X_val,Y_val = get_CV_splits(X,Y,nfolds,rng_key)

    @jax.jit
    def loop(dummy, i:Scalar):
        X_tr,Y_tr,X_val,Y_val,theta=dummy
        l=learner.loss(X_tr[:,:,i],Y_tr[:,:,i],X_val[:,:,i],Y_val[:,:,i],jnp.exp(theta))
        return (X_tr,Y_tr,X_val,Y_val,theta),l

    @jax.jit
    def CV_loss(theta:Array,
                X_tr:Array=X_tr,
                Y_tr:Array=Y_tr,
                X_val:Array=X_val,
                Y_val:Array=Y_val):
        dim=X_tr.shape[2]
        _,loss_array=jax.lax.scan(loop, (X_tr,Y_tr,X_val,Y_val,theta), jnp.arange(dim))
        l=jnp.sum(loss_array)
        return l

    @jax.jit
    def one_step(dummy,i: Scalar):
        opt_state, params, step_size = dummy
        optimizer = optax.adam(learning_rate=step_size)

        grads = grad(CV_loss)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params_next = optax.apply_updates(params, updates)

        dummy_next = (new_opt_state, params_next, step_size)
        return dummy_next, params_next

    dummy=(opt_state, params, step_size)

    _,params_list=jax.lax.scan(one_step,dummy,jnp.arange(step_num))

    return params_list[-1]
    

def optimise_lambda(learner,
                    X:Array,
                    Y:Array,
                    sigma_X:Array,
                    step_num:int = 100,
                    nfolds:int = 3,
                    step_size:float = 1e-2, 
                    subsampling:bool=False,
                    subsamples:int=1000,
                    rng_key:Array=None):
    """
        Gradient-based tuning of functional hyperparameters by K-fold
        kernel-RKHS CV risk.

        - For each epoch, we iterate K folds:
            optionally subsample the training fold,
            fit the functional on the training fold,
            compute RKHS loss on the validation fold using S = linear_weights,
            accumulate mean CV loss,
            backprop once per epoch.
    """
    params = jnp.ones((1,))

    optimizer = optax.adam(learning_rate=step_size)
    opt_state = optimizer.init(params)

    if subsampling:
        X,Y=subsample_train_fold(X,Y,subsamples,rng_key)

    X_tr,Y_tr,X_val,Y_val = get_CV_splits(X,Y,nfolds,rng_key)
    
    @jax.jit
    def loop(dummy, i:Scalar):
        X_tr,Y_tr,X_val,Y_val,theta=dummy
        l=learner.loss(X_tr[:,:,i],Y_tr[:,:,i],X_val[:,:,i],Y_val[:,:,i],theta)
        return (X_tr,Y_tr,X_val,Y_val,theta),l

    @jax.jit
    def CV_loss(theta:Array,
                sigma_X:Array=sigma_X,
                X_tr:Array=X_tr,
                Y_tr:Array=Y_tr,
                X_val:Array=X_val,
                Y_val:Array=Y_val):
        dim=X_tr.shape[2]
        param=jnp.append(jnp.exp(theta),sigma_X)
        _,loss_array=jax.lax.scan(loop, (X_tr,Y_tr,X_val,Y_val,param), jnp.arange(dim))
        l=jnp.sum(loss_array)
        return l

    @jax.jit
    def one_step(dummy,i: Scalar):
        opt_state, params, step_size = dummy
        optimizer = optax.adam(learning_rate=step_size)

        grads = grad(CV_loss)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        params_next = optax.apply_updates(params, updates)

        dummy_next = (new_opt_state, params_next, step_size)
        return dummy_next, params_next

    dummy=(opt_state, params, step_size)

    _,params_list=jax.lax.scan(one_step,dummy,jnp.arange(step_num))

    return params_list[-1]

def learn_hyperparams(A_tr:Array,
                      M_tr:Array,
                      X_tr:Array,
                      Y_tr:Array,
                      subsample:bool=True):
    """automated loop for learning kernel lengthscales and regression regularisation hyperparameters"""
    n_tr=Y_tr.shape[0]
    d_m=M_tr.shape[1]
    d_x=X_tr.shape[1]
    
    AX=jnp.concatenate((A_tr,X_tr),axis=1)
    AMX=jnp.concatenate((A_tr,M_tr,X_tr),axis=1)

    kernel=gaussian_kernel(ard_median_heuristic(Y_tr))
    jax_ridge = KRR(kernel)
    hyperparams=jnp.exp(optimise(jax_ridge,AMX,Y_tr,subsampling=subsample))
    lambda_y=hyperparams[0]*jnp.ones((1,))
    sigma_A=hyperparams[1]*jnp.ones((1,))
    sigma_M=hyperparams[2:2+d_m]*jnp.ones((d_m,))
    sigma_X=hyperparams[2+d_m:]*jnp.ones((d_x,))

    sigma_AX=jnp.concatenate((sigma_A,sigma_X),axis=0)
    sigma_AMX=jnp.concatenate((sigma_A,sigma_M,sigma_X),axis=0)

    A_kernel = gaussian_kernel(sigma_A)
    K_AA=A_kernel.make_distance_matrix(A_tr,A_tr)
    M_kernel = gaussian_kernel(sigma_M)
    K_MM=M_kernel.make_distance_matrix(M_tr,M_tr)
    X_kernel = gaussian_kernel(sigma_X)
    K_XX=X_kernel.make_distance_matrix(X_tr,X_tr)
    
    jax_M_ridge=KRR(M_kernel)
    lambda_m=jnp.exp(optimise_lambda(jax_M_ridge,AX,M_tr,sigma_AX,subsampling=subsample))
    
    return lambda_m,lambda_y,sigma_A,sigma_M,sigma_X


"""Functions below this line are not used in the script
-----------------------------------------------------------------------------------------------------------------"""

def find_best_parameter(X, Y, sigma, reg_grid=[1e1,1e0,1e-1,1e-2,1e-3,1e-4], num_cv=3,Y_sigma=None):
    if Y_sigma is None:
        Y_sigma=ard_median_heuristic(Y)
    kernel=gaussian_kernel(Y_sigma)
    jax_ridge = KRR(kernel)
    lmbda=reg_grid[0]*jnp.ones((1,))
    theta_array=jnp.append(lmbda,sigma)
    for i in range(1,len(reg_grid)):
        l = reg_grid[i]*jnp.ones((1,))
        theta = jnp.append(l,sigma)
        theta_array = jnp.vstack((theta_array,theta))

    best_param=CV_gridsearch(jax_ridge,X,Y,num_cv,theta_array)
    return best_param


def CV_gridsearch(learner,X,Y,nfolds,param_list,subsampling=False,subsamples=1000,rng_key=None):
    #learner must implement learner.loss(X-train,Y-train,X-test,Y-test,params)

    if subsampling:
        X,Y=subsample_train_fold(X,Y,subsamples,rng_key)

    X_tr,Y_tr,X_val,Y_val = get_CV_splits(X,Y,nfolds,rng_key)

    def CV_loss(X_tr,Y_tr,X_val,Y_val,theta):
        loss_array=vmap(learner.loss, (2,2,2,2,None))(X_tr,Y_tr,X_val,Y_val,theta)
        #print(loss_array)
        return loss_array

    def CV_loss_alt(X_tr,Y_tr,X_val,Y_val,theta):
        l = 0
        for i in range(X_tr.shape[2]):
            l+=learner.loss(X_tr[:,:,i],Y_tr[:,:,i],X_val[:,:,i],Y_val[:,:,i],theta)
        return l
    #losses=vmap(CV_loss_alt, (None,None,None,None,0))(X_tr,Y_tr,X_val,Y_val,param_list)
    #print(losses)

    loss_list=[]
    for i in range(param_list.shape[0]):
        #print(CV_loss(X_tr,Y_tr,X_val,Y_val,param_list[i,:]))
        loss_list.append(CV_loss_alt(X_tr,Y_tr,X_val,Y_val,param_list[i,:]))

    losses = jnp.stack(loss_list)
    best_param_index = jnp.argmin(losses)

    best_params = param_list[best_param_index,:]
    
    return best_params