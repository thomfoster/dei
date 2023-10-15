import scipy
import numpy as np


def log_likelihood(mu, cov, y, sigma_squared=1e-10):
    # Note this is the same computation as the log probs of the data given the hyperparams
    # add conditioning to cov
    cov = cov + + sigma_squared * np.eye(cov.shape[0])
    L = scipy.linalg.cholesky(cov, lower=True)
    alpha = scipy.linalg.solve(L.T, np.linalg.solve(L, (y - mu)))
    
    A = -0.5 * (y - mu).T @ alpha
    B = -np.trace(L)
    C = -0.5 * mu.shape[0] * np.log(2 * np.pi)

    return A + B + C


def GP(X_d, y_d, X_star, mu_func, k_func, sigma_squared=0.001, method='cholesky', verbose=True):
    assert X_d.shape[0] == y_d.shape[0]
    if verbose:
        print(f"Constructing GP from {X_d.shape[0]} training points and {X_star.shape[0]} eval locations.")

    # compute key matrices
    k_d_d = k_func(X_d, X_d)
    k_d_star = k_func(X_d, X_star)
    k_star_star = k_func(X_star, X_star)

    mu_d = np.array([mu_func(X_d[i]) for i in range(X_d.shape[0])])
    mu_star = np.array([mu_func(X_star[i]) for i in range(X_star.shape[0])])

    # add conditioning to k_d_d
    k_d_d = k_d_d + sigma_squared * np.eye(k_d_d.shape[0])

    if method == 'scipy_solve':
        solved = scipy.linalg.solve(k_d_d, k_d_star, assume_a='pos').T
        new_mu = mu_star + solved @ (y_d - mu_d)
        new_cov = k_star_star - (solved @ k_d_star)

        prior_mismatch = -0.5 * (y_d - mu_d).T @ scipy.linalg.solve(k_d_d, (y_d - mu_d), assume_a='pos')
        model_complexity = -0.5 * scipy.log(scipy.linalg.det(k_d_d))
        constant = -0.5 * X_d.shape[0] * np.log(2 * np.pi)
        log_p = prior_mismatch + model_complexity + constant

    elif method == 'cholesky':
        L = scipy.linalg.cholesky(k_d_d, lower=True)
        alpha = scipy.linalg.solve(L.T, np.linalg.solve(L, (y_d - mu_d)))
        new_mu = mu_star + k_d_star.T @ alpha
        v = np.linalg.solve(L, k_d_star)
        new_cov = k_star_star - (v.T @ v)

        prior_mismatch = -0.5 * (y_d - mu_d).T @ alpha
        model_complexity = -np.trace(L)
        constant = -0.5 * X_d.shape[0] * np.log(2 * np.pi)
        log_p = prior_mismatch + model_complexity + constant
        
    else:
        raise RuntimeError(f"Method: {method} not recognised.")
    

    return new_mu, new_cov, log_p  # mean, covariance
