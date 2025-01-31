import numpy as np
import scipy
import scipy.linalg
import cvxpy as cp

from tqdm import tqdm
from scipy.optimize import minimize


def map_estimation(
    B: np.ndarray,
    Sigma: np.ndarray,
    Theta: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    lambda0: float,
    lambda1: float,
    convergence_criterion: float = 0.05,
    num_iter: int = 500,
):
    """
    Get MAP estimates for B, Sigma and Theta.

    Args:
        B (np.tensor): size (G*K)
        Sigma (np.tensor): size (G)
        Theta (np.tensor): size (K)
        Y (np.tensor): size (G*n)
        alpha (float)
        lambda0 (float)
        lambda1 (float)
        num_var(int): G
        num_obs (int): n
        num_factors (int): K
        convergence_criterion (float)
        forced_stop (int)

    Returns:
        B (np.tensor): size (G*K)
        Sigma (np.tensor): size (G)
        Theta (np.tensor): size (K)
    """

    num_var, num_obs = Y.shape
    _, num_factors = B.shape

    error = np.inf
    B_star_old = np.zeros((num_factors, num_var))

    history = {}

    with tqdm(total=num_iter, desc="PXL-EM", unit="iter") as pbar:
        for _ in range(num_iter):

            # E-Step
            Omega, M = update_latent_features(B, Sigma, Y, num_factors)
            Gamma = get_latent_indicators(B, Theta, lambda0, lambda1)

            # M-Step
            ## Set new variables
            Y_tilde = np.vstack(
                [Y.T, np.zeros((num_factors, num_var))]
            )  # size ((K+n)*G)
            Omega_tilde = get_Omega_tilde(Omega, M, num_obs)
            B_star = np.zeros((num_factors, num_var))
            Theta = np.zeros(num_factors)
            A = update_rotation(Omega, M, num_obs)
            ## Update
            for j in range(num_var):
                B_star[:, j] = update_loading(
                    Y_tilde, Omega_tilde, Sigma, Gamma, num_factors, j
                )
                Sigma[j] = update_variance(Y_tilde, Omega_tilde, B_star, num_obs, j)

            Theta = update_sparsity(Gamma, alpha)
            # Rotation Step
            B = rotation(B_star.T, A)

            # Update distance
            error = infinite_norm_distance(B_star, B_star_old)
            B_star_old = B_star

            pbar.set_postfix({"error": f"{error:.5f}"})
            pbar.update(1)

            if error < convergence_criterion:
                break

            history[error] = {"B": B, "Sigma": Sigma, "Theta": Theta, "Omega": Omega}

    return history


######## E-Step
def update_latent_features(B, Sigma, Y, num_factors):
    """
    Extract the latent features mean.

    Args:
        B (np.tensor): size (G*K)
        Sigma (np.tensor): size (G)
        Y (np.tensor): size (G*n)
        num_factors (int): K

    Returns:
        Omega (np.tensor): size (K*n)
        M (np.tensor): size (K*K)
    """

    precision = np.eye(num_factors) + B.T @ np.diag(1 / Sigma) @ B
    M = np.linalg.inv(precision)
    Omega = M @ B.T @ np.diag(1 / Sigma) @ Y

    return Omega, M


def get_latent_indicators(B, Theta, lambda0, lambda1):
    """
    Extract the latent indicators means.

    Args:
        B (np.tensor): size (G*K)
        Theta (np.tensor): size (K)
        lambda0 (float)
        lambda1 (float)

    Returns:
        gamma (np.tensor): size (G*K)
    """
    return (lambda1 * np.exp(-lambda1 * np.abs(B)) * Theta) / (
        lambda0 * np.exp(-lambda0 * np.abs(B)) * (1 - Theta)
        + lambda1 * np.exp(-lambda1 * np.abs(B)) * Theta
    )


######## M-Step


def get_Omega_tilde(Omega, M, num_obs):
    """
    Get Omega_tilde = (Omega, sqrt(n)M_L).

    Args:
        Omega (np.tensor): size (K*n)
        M (np.tensor): size (K*K)
        num_obs (int)

    Returns:
        Omega_tilde (np.tensor): size ((K+n)*K)
    """
    M_L = scipy.linalg.cholesky(M, lower=True)
    low = np.sqrt(num_obs) * M_L
    return np.vstack([Omega.T, low])


def update_loading(Y_tilde, Omega_tilde, Sigma, Gamma, num_factor, j):
    """
    Computes the updated value of beta_j_star using the iterative update formula.

    beta_j_star = (|z| - sigma_j * lambda_jk (A_L^-1) / n) * sign(z) - z_k+

    Args:
        Y_tilde (np.tensor): size ((K+n)*G)
        Omega_tilde (np.tensor): size ((K+n)*K)
        B (np.tensor): size (G*K)
        Sigma (np.tensor): size (G)
        gamma (np.tensor): size (G*K)
        A (np.tensor): size (K*K)
        num_factors (int): K
        j (int)

    Returns:
        B_j_star (np.tensor): size(K)
    """

    def objective(B_j_star):
        return np.sum((Y_tilde[:, j] - Omega_tilde @ B_j_star) ** 2) + 2 * Sigma[
            j
        ] ** 2 * np.sum(np.abs(B_j_star * Gamma[j, :]))

    B_j_star = minimize(objective, np.random.rand(num_factor)).x

    return B_j_star


def update_variance(Y_tilde, Omega_tilde, B_star, num_obs, j):
    """
    Update the jth diagonal coefficient of Sigma.

    Args:
        Y_tilde (np.tensor): size ((K+n)*G)
        Omega_tilde (np.tensor): size ((K+n)*K)
        beta_j_star (np.tensor): size(K)
        num_obs (int)
        j (int)

    Returns:
        sigma_j (float)
    """
    return (np.sum((Y_tilde[:, j] - Omega_tilde @ B_star[:, j]) ** 2) + 1) / (
        num_obs + 1
    )


def update_rotation(Omega, M, num_obs):
    return (Omega @ Omega.T) / num_obs + M


def update_sparsity(Gamma, alpha):
    K = Gamma.shape[1]
    theta = cp.Variable(K, nonneg=True)

    # Objective function and constraints
    objective = cp.Maximize(
        cp.sum(cp.multiply(Gamma, cp.log(theta)))
        + cp.sum(cp.multiply(1 - Gamma, cp.log(1 - theta)))
        + (1 - alpha) * cp.log(theta[-1])
    )
    constraints = [theta[k] - theta[k - 1] <= 0 for k in range(1, K)] + [theta <= 1]

    # Solve the problem
    cp.Problem(objective, constraints).solve()

    return theta.value


######## Rotation Step


def rotation(B_star, A):
    """
    Rotates B_star to get B.

    Args:
        B_star (np.tensor): size (G*(K+n))
        A (np.tensor): size (K*K)

    Returns:
        B (np.tensor): size (G*K)
    """
    A_L = scipy.linalg.cholesky(A, lower=True)  # size (K*K)

    return B_star @ A_L


######## Convergence criterion


def infinite_norm_distance(A, B):
    """
    Computes the distance between two matrices in the infinite norm.

    Args:
        A (np.Tensor): First matrix of size [m, n].
        B (np.Tensor): Second matrix of size [m, n].

    Returns:
        float: The distance between A and B in the infinite norm.
    """
    # Ensure A and B have the same shape
    assert A.shape == B.shape, "Matrices A and B must have the same shape"

    # Compute the element-wise absolute difference
    diff = np.abs(A - B)

    # Compute the row sums
    row_sums = np.sum(diff, axis=1)

    return np.max(row_sums).item()
