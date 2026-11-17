from typing import Any, Callable, Type, Optional
from typing_extensions import TypeAlias, Self
import jax
import jax.numpy as jnp
from jax import vmap,grad
import numpy as np
import abc
from functools import partial
from jax.tree_util import tree_map
from utils import base_kernel, Array



def _l2_norm_squared(x: Array) -> Array:
    #helper function for use in kernels
    return jnp.sum(jnp.square(x))

class gaussian_kernel(base_kernel):
    sigma: Array

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-0.5 * _l2_norm_squared(jnp.divide((x - y),self.sigma)))
    
    def grad_x(self, x: Array, y: Array) -> Array:
        #∇_x k
        dvec = x - y
        r2=_l2_norm_squared(x-y)
        inv_sigma2 = jnp.pow(self.sigma,-2)
        kxy = self.__call__(x,y)
        return -(jnp.multiply(dvec, inv_sigma2)) * kxy

    def grad_y(self, x: Array, y: Array) -> Array:
        #∇_y k
        dvec = x - y
        r2=_l2_norm_squared(x-y)
        inv_sigma2 = jnp.pow(self.sigma,-2)
        kxy = self.__call__(x,y)
        return +(jnp.multiply(dvec, inv_sigma2)) * kxy

    def grad_xy(self, x: Array, y: Array) -> Array:
        #∇_x ∇_y k
        r2=_l2_norm_squared(x-y)
        d = x.shape[0]
        inv_sigma2 = 1.0 / (self.sigma * self.sigma)
        inv_sigma4 = inv_sigma2 * inv_sigma2
        kxy = self.__call__(x,y)
        return (d * inv_sigma2 - r2 * inv_sigma4) * kxy

class IMQ_kernel(base_kernel):
    sigma: Array

    def __call__(self, x: Array, y:Array) -> Array:
        #k(x,y) = (1 + 0.1^(−2)|x−y|^2)^(-1/2)
        l2=_l2_norm_squared(x-y)
        return jnp.power((1.+0.1**(-2)*l2),-0.5)
    
    def grad_x(self, x: Array, y: Array) -> Array:
        #∇_x k =  -(0.1^(−2))(x−y)*(1 + 0.1^(−2)|x−y|^2)^(-3/2)
        l2=_l2_norm_squared(x-y)
        return -1.*0.1**(-2)*(x-y)*jnp.power((1.+0.1**(-2)*l2),-1.5)

    def grad_y(self, x: Array, y: Array) -> Array:
        #∇_y k =  (0.1^(−2))(x−y)*(1 + 0.1^(−2)|x−y|^2)^(-3/2)
        l2=_l2_norm_squared(x-y)
        return 0.1**(-2)*(x-y)*jnp.power((1.+0.1**(-2)*l2),-1.5)

    def grad_xy(self, x: Array, y: Array) -> Array:
        #∇_x ∇_y k = d * 0.1^(-2)*(1 + 0.1^(−2)|x−y|^2)^(-3/2) - 3*0.1^(-4)(1 + 0.1^(−2)|x−y|^2)^(-5/2)
        d = x.shape[0]
        l2=_l2_norm_squared(x-y)
        return d * 0.1**(-2)*jnp.power((1.+0.1**(-2)*l2),-1.5) - 3*0.1**(-4)*l2*jnp.power((1.+0.1**(-2)*l2),-2.5)