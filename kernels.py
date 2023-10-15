import numpy as np
import scipy

def combine_add(k_func_1, k_func_2, alpha=0.5):
    def new_k_func(xa, xb):
        return alpha * k_func_1(xa, xb) + (1 - alpha) * k_func_2(xa, xb)
    return new_k_func

def combine_product(k_func_1, k_func_2):
    def new_k_func(xa, xb):
        return k_func_1(xa, xb) * k_func_2(xa, xb)
    return new_k_func

# def get_numpy_exponentiated_quadratic(lam, omega):
#     print('Warning! This is legacy code that is incorrectly vectorised')
#     # Define the exponentiated quadratic 
#     def exponentiated_quadratic(x1, x2):
#         dist = np.linalg.norm(x1 - x2, axis=-1) / omega
#         print(dist.shape)
#         return lam**2 * np.exp(-0.5 * dist**2)
#     return exponentiated_quadratic

def get_exponentiated_quadratic(lam, omega, dims=None):
    def exponentiated_quadratic(xa, xb):
        """Exponentiated quadratic  with σ=1"""
        # L2 distance (Squared Euclidian)
        if dims is not None:
            xa, xb = xa[:, dims], xb[:, dims]
        if np.ndim(xa) == 1:
            xa, xb = np.expand_dims(xa, 1), np.expand_dims(xb, 1)
        sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') / (omega**2)
        return (lam**2) * np.exp(sq_norm)
    return exponentiated_quadratic

def get_rational_quadratic(lam, omega, alpha):
    def rational_quadratic(xa, xb):
        """Exponentiated quadratic  with σ=1"""
        # L2 distance (Squared Euclidian)
        if np.ndim(xa) == 1:
            xa, xb = np.expand_dims(xa, 1), np.expand_dims(xb, 1)
        adjusted_dist =  0.5 * alpha**-1 * omega**-2 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
        rational_norm = (1 + adjusted_dist)**-alpha
        return (lam**2) * rational_norm
    return rational_quadratic
    
def get_matern_1_2(lam, omega):
    def matern(xa, xb):
        # L2 distance (Squared Euclidian)
        if np.ndim(xa) == 1:
            xa, xb = np.expand_dims(xa, 1), np.expand_dims(xb, 1)
        dist =  scipy.spatial.distance.cdist(xa, xb, 'euclidean') / omega
        return (lam**2)*np.exp(-dist)
    return matern

def get_matern_3_2(lam, omega):
    def matern(xa, xb):
        # L2 distance (Squared Euclidian)
        if np.ndim(xa) == 1:
            xa, xb = np.expand_dims(xa, 1), np.expand_dims(xb, 1)
        dist =  scipy.spatial.distance.cdist(xa, xb, 'euclidean') / omega
        return (lam**2) * (1 + np.sqrt(3)*dist) * np.exp(-np.sqrt(3)*dist)
    return matern

def get_periodic(lam, omega, p):
    def periodic(xa, xb):
        # L2 distance (Squared Euclidian)
        if np.ndim(xa) == 1:
            xa, xb = np.expand_dims(xa, 1), np.expand_dims(xb, 1)
        dist =  np.pi * scipy.spatial.distance.cdist(xa, xb, 'euclidean') / p
        return (lam**2) * np.exp(-2 * (np.sin(dist)**2) / omega)
    return periodic

def get_affine(alpha, beta, gamma):
    def affine(xa, xb):
        if np.ndim(xa) == 1:
            xa, xb = np.expand_dims(xa, 1), np.expand_dims(xb, 1)
        return alpha**2 + (beta**2 * (xa - gamma) * (xb - gamma).T )
    return affine