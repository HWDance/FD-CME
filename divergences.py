import jax
import optax
import jax.numpy as jnp
from jax import vmap,grad,random
from flax import struct
import numpy as np
from functools import partial
from typing import Any, Callable, Type, Optional
from typing_extensions import TypeAlias, Self
from utils import Array, Scalar, base_kernel

class mmd_fixed_target:
    #based on MMD gradient flow implementation from https://github.com/hudsonchen/DrMMD/blob/main/kwgflows/divergences/mmd.py
    def __init__(self, kernel:base_kernel):
        self.kernel = kernel

    def pre_compute(self, X:Array):
        #X are samples from the target distribution, so they remain constant throughout
        self.X = X

    def get_witness_function(self, z:Array, Y:Array) -> Scalar:
        z = z[None, :]
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (K_zY.mean(1) - K_zX.mean(1)).squeeze()

    def get_first_variation(self, Y:Array) -> Callable:
        return partial(self.get_witness_function, Y=Y)

    def __call__(self, Y:Array) -> Scalar: # mmd^2
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()


class mmd_dr_embedding:
    #based on mmd_fixed_targets above, but using counterfactual mean embeddings instead of samples from target
    #for plugin version, make X_te and dr an array of zeros, use the plugin vector instead of eif
    def __init__(self, kernel:base_kernel, dr:Array, eif:Array):
        self.kernel = kernel
        self.eif = eif
        self.dr = dr# / dr.shape[0]

    def pre_compute(self, X_tr:Array,X_te:Array):
        
        self.X_tr = X_tr
        self.X_te= X_te

    def get_witness_function(self, z:Array, Y:Array) -> Scalar:
        z = z[None, :]
        K_zX = jnp.matmul(self.kernel.make_distance_matrix(z, self.X_tr),self.eif) + jnp.matmul(self.kernel.make_distance_matrix(z, self.X_te),self.dr)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (K_zY.mean(1) - K_zX.mean(1)).squeeze()

    def get_first_variation(self, Y:Array) -> Callable:
        return partial(self.get_witness_function, Y=Y)

    def __call__(self, Y:Array) -> Scalar: # mmd^2

        K_YY = self.kernel.make_distance_matrix(Y, Y)

        K_tr_tr=self.kernel.make_distance_matrix(self.X_tr,self.X_tr)
        K_tr_te=self.kernel.make_distance_matrix(self.X_tr,self.X_te)
        K_te_te=self.kernel.make_distance_matrix(self.X_te,self.X_te)

        K_tr_Y=self.kernel.make_distance_matrix(self.X_tr,Y)
        K_te_Y=self.kernel.make_distance_matrix(self.X_te,Y)
        first_term = self.eif.T @ K_tr_tr @ self.eif + self.dr.T @ K_te_te @ self.dr + self.eif.T @ K_tr_te @ self.dr + self.dr.T @ K_tr_te.T @ self.eif
        second_term = self.dr.T @ K_te_Y + K_te_Y.T @ self.dr + self.eif.T @ K_tr_Y + K_tr_Y.T @ self.eif
        third_term = K_YY
        return first_term.mean() - second_term.mean() + third_term.mean()

class KSD_dr:
    #KSD between distribution and counterfactual mean embedding
    #if using plugin estimator, set dr and X_te to be array of zeros
    def __init__(self, distribution:Learner, kernel:base_kernel, dr:Array, EIF:Array):
        self.kernel = kernel
        self.distribution = distribution
        self.dr = dr# / dr.shape[0]
        self.eif = EIF

    def make_distance_matrix(self, X: Array, Y: Array, theta:Array) -> Array:
        #modified from base_kernel class to include theta
        return vmap(vmap(type(self).__call__, (None, None, 0, None)), (None, 0, None, None))(self, X, Y, theta)

    def __call__(self, x:Array, y:Array, theta:Array) -> Scalar:

        #scores of the parametric distribution
        sx = self.distribution.score(theta=theta,x=x)
        sy = self.distribution.score(theta=theta,x=y)

        #base kernel and its gradients
        k_xy = self.kernel.__call__(x, y)
        grad_x_k_xy = self.kernel.grad_x(x,y)
        grad_y_k_xy = self.kernel.grad_y(x,y)
        grad_xy_k_xy = self.kernel.grad_xy(x,y)
        
        term_1=jnp.dot(sx,sy)*k_xy
        term_2=jnp.dot(sy, grad_x_k_xy)
        term_3=jnp.dot(sx, grad_y_k_xy)
        term_4=grad_xy_k_xy
        return  term_1+ term_2 + term_3 + term_4


    def pre_compute(self, X_tr:Array, X_te:Array):
        self.X_tr=X_tr
        self.X_te=X_te
        
    def loss(self,theta:Array) -> Scalar:
        #KSD calculation
        K_tr_tr=self.make_distance_matrix(self.X_tr,self.X_tr, theta)
        K_tr_te=self.make_distance_matrix(self.X_tr,self.X_te, theta)
        K_te_te=self.make_distance_matrix(self.X_te,self.X_te, theta)
        first_term = self.eif.T @ K_tr_tr @ self.eif + self.dr.T @ K_te_te @ self.dr + self.eif.T @ K_tr_te @ self.dr + self.dr.T @ K_tr_te.T @ self.eif
        return first_term.mean()

class KSD:
    #KSD between distribution and sample X
    def __init__(self, distribution:Learner, kernel:base_kernel):
        self.kernel = kernel
        self.distribution = distribution

    def make_distance_matrix(self, X: Array, Y: Array, theta:Array) -> Array:
        #modified from base_kernel class to include theta
        return vmap(vmap(type(self).__call__, (None, None, 0, None)), (None, 0, None, None))(self, X, Y, theta)

    def __call__(self, x:Array, y:Array, theta:Array) -> Scalar:
        #compute scores
        sx = self.distribution.score(theta=theta,x=x)
        sy = self.distribution.score(theta=theta,x=y)

        #base kernel and its gradients
        k_xy = self.kernel.__call__(x, y)
        grad_x_k_xy = self.kernel.grad_x(x,y)
        grad_y_k_xy = self.kernel.grad_y(x,y)
        grad_xy_k_xy = self.kernel.grad_xy(x,y)
        
        term_1=jnp.dot(sx,sy)*k_xy
        term_2=jnp.dot(sy, grad_x_k_xy)
        term_3=jnp.dot(sx, grad_y_k_xy)
        term_4=grad_xy_k_xy
        return  term_1+ term_2 + term_3 + term_4

    def pre_compute(self, X:Array):
        self.X=X
        
    def loss(self,theta:Array) -> Scalar:
        #KSD calculation
        K_XX=self.make_distance_matrix(self.X,self.X, theta)
        return K_XX.mean()