from sklearn.metrics import pairwise_kernels, pairwise_distances
import numpy as np
from scipy.optimize import minimize
from utils import Array,Scalar
rng = np.random.default_rng(0)

def kernel_herding_EIF(Y_te,Y_tr,EIF_estimate,dr_array,n):
    """kernel herding for one-step estimator"""
    sigma=np.median(pairwise_distances(Y_tr)) ** 2 + 1e-8
    d=Y_tr.shape[1]
    #guess for location of sample
    if d == 1:
        y0 = rng.standard_normal()
    else:
        y0 = rng.standard_normal(size=(d,))
    n_te=Y_te.shape[0]

    #first minimisation
    res = minimize(
        lambda y: -np.dot(pairwise_kernels(np.atleast_2d(y),Y_te,metric="rbf",gamma=1.0 / (sigma)).ravel(), dr_array)
        -(pairwise_kernels(np.atleast_2d(y),Y_tr,metric="rbf",gamma=1.0 / (sigma))@EIF_estimate).item(),
        y0,
        method="BFGS",
        options={"gtol": 1e-6, "disp": False},
    )
    y1 = res.x
    y_list = [y1]

    #sequential minmisation in kernel herding
    for t in range(2, n + 1):
        y_hist = np.array(y_list)
        res = minimize(
            lambda y: -np.dot(pairwise_kernels(np.atleast_2d(y),Y_te,metric="rbf",gamma=1.0 / (sigma)).ravel(), dr_array)
            -(pairwise_kernels(np.atleast_2d(y),Y_tr,metric="rbf",gamma=1.0 / (2 * sigma))@EIF_estimate).item()
            + np.mean(pairwise_kernels(y_hist,np.atleast_2d(y),metric="rbf",gamma=1.0 / (sigma))
            ),
            y0,
            method="BFGS",
            options={"gtol": 1e-6, "disp": False},
        )
        yt = res.x
        y_list.append(yt)
    return np.array(y_list)

def kernel_herding_plugin(Y_tr,plugin_estimate,n):
    """kernel herding for plug-in estimator"""
    sigma=np.median(pairwise_distances(Y_tr)) ** 2 + 1e-8

    d=Y_tr.shape[1]
    #guess for location of sample
    if d == 1:
        y0 = rng.standard_normal()
    else:
        y0 = rng.standard_normal(size=(d,))


    #first minimisation
    res = minimize(
        lambda y: -np.dot(pairwise_kernels(Y_tr,np.atleast_2d(y),metric="rbf",gamma=1.0 / (sigma)).T, plugin_estimate),
        y0,
        method="BFGS",
        options={"gtol": 1e-6, "disp": False},
    )
    y1 = res.x
    y_list = [y1]

    #sequential minimisation for kernel herding
    for t in range(2, n + 1):
        y_hist = np.array(y_list)
        res = minimize(
            lambda y: -np.dot(pairwise_kernels(Y_tr,np.atleast_2d(y),metric="rbf",gamma=1.0 / (sigma)).T, plugin_estimate)
            + np.mean(pairwise_kernels(y_hist,np.atleast_2d(y),metric="rbf",gamma=1.0 / (sigma))
            ),
            y0,
            method="BFGS",
            options={"gtol": 1e-6, "disp": False},
        )
        yt = res.x
        y_list.append(yt)
    return np.array(y_list)