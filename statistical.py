# imports
import numpy as np
from scipy.stats import norm

# single mode normal fit to data
def normal_params(data):
    """
    Compute the mean and variance of a normal distribution fit to the input data.

    Parameters:
    data (numpy.ndarray): Array of input data points.

    Returns:
    tuple: Mean and variance of the fitted normal distribution.
    """
    mean = np.mean(data)
    variance = np.var(data)
    return mean, variance

# Bimodal mixture of two normal distributions fit to data using the EM algorithm
def bimodal_normal_params(data, max_iter=100, tol=1e-6):
    """
    Fit a bimodal mixture of two normal distributions to the input data using the EM algorithm.

    Parameters:
    data (numpy.ndarray): Array of input data points.
    max_iter (int): Maximum number of iterations for the EM algorithm.
    tol (float): Convergence tolerance for the EM algorithm.

    Returns:
    tuple: Means, variances, and mixing coefficients of the fitted bimodal mixture.
    """
    # Step 1: Initialize parameters
    n = len(data)
    np.random.seed(0)
    
    # Randomly initialize means, variances, and mixing coefficients
    mu1, mu2 = np.random.choice(data, 2)
    var1, var2 = np.var(data), np.var(data)
    pi1, pi2 = 0.5, 0.5  # Start with equal mixing coefficients

    for _ in range(max_iter):
        # Step 2: E-step - Calculate responsibilities (posterior probabilities)
        r1 = pi1 * norm.pdf(data, mu1, np.sqrt(var1))
        r2 = pi2 * norm.pdf(data, mu2, np.sqrt(var2))
        
        total_r = r1 + r2
        gamma1 = r1 / total_r
        gamma2 = r2 / total_r

        # Step 3: M-step - Update parameters based on responsibilities
        N1, N2 = np.sum(gamma1), np.sum(gamma2)
        
        # Update means
        new_mu1 = np.sum(gamma1 * data) / N1
        new_mu2 = np.sum(gamma2 * data) / N2
        
        # Update variances
        new_var1 = np.sum(gamma1 * (data - new_mu1)**2) / N1
        new_var2 = np.sum(gamma2 * (data - new_mu2)**2) / N2
        
        # Update mixing coefficients
        new_pi1 = N1 / n
        new_pi2 = N2 / n

        # Check for convergence
        if (np.abs(new_mu1 - mu1) < tol and np.abs(new_mu2 - mu2) < tol and
            np.abs(new_var1 - var1) < tol and np.abs(new_var2 - var2) < tol and
            np.abs(new_pi1 - pi1) < tol and np.abs(new_pi2 - pi2) < tol):
            break

        # Update parameters for the next iteration
        mu1, mu2 = new_mu1, new_mu2
        var1, var2 = new_var1, new_var2
        pi1, pi2 = new_pi1, new_pi2

    return (mu1, var1, pi1), (mu2, var2, pi2)

# Gaussian kernel function
def gaussian_kernel(x, xi, h):
    """
    Compute the Gaussian kernel value for a given point x and data point xi.

    Parameters:
    x (float): The point at which to evaluate the kernel.
    xi (float): A data point from the dataset.
    h (float): Bandwidth parameter for the kernel.

    Returns:
    float: Value of the Gaussian kernel at x.
    """
    return (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-0.5 * ((x - xi) / h) ** 2)

# KDE computation function
def gaussian_kde(data, h=10):
    """
    Compute the Kernel Density Estimate (KDE) using a Gaussian kernel.

    Parameters:
    data (numpy.ndarray): Array of input data points.
    h (float): Bandwidth for the Gaussian kernel (default is 10).

    Returns:
    numpy.ndarray -> numpy.ndarray: KDE function that can be evaluated at any point.
    """
    def kde(x):
        """
        Compute the KDE at point x by averaging the contributions of each data point.

        Parameters:
        x (float): The point at which to evaluate the KDE.

        Returns:
        float: KDE estimate at x.
        """
        return np.mean([gaussian_kernel(x, xi, h) for xi in data])
    
    return kde

# Finds local maxima of a list of values
def find_mode_locations(values):
    """
    Find the indices where the function goes from increasing to decreasing,
    with the condition that the 5 values to the left of the index must be strictly increasing,
    and the 5 values to the right of the index must be strictly decreasing.

    Parameters:
    values (list or numpy.ndarray): A list or array of numeric values.

    Returns:
    list: A list of indices where the function transitions from increasing to decreasing.
    """
    mode_locations = []
    window_size = 5  # Number of neighbors to check on each side

    # Iterate through the list, skipping the first and last 5 points (due to the window)
    for i in range(window_size, len(values) - window_size):
        # Check if the left side is strictly increasing
        if (values[i - 5] < values[i - 4] < values[i - 3] < values[i - 2] < values[i - 1] < values[i]):
            # Check if the right side is strictly decreasing
            if (values[i] > values[i + 1] > values[i + 2] > values[i + 3] > values[i + 4] > values[i + 5]):
                mode_locations.append(i)

    return mode_locations

# get performance out of predictions
def kde_predictions(dist_pred):
    """
    Gets performance of a KDE approximation to simulated distribution.

    Parameters:
    dist_pred (list or numpy.ndarray): A list or array of numeric values.

    Returns:
    arr: array of predicted modes.
    """
    x_range = np.arange(-70, 205, 1)
    kde_func = gaussian_kde(dist_pred, h=5)
    density_func = [kde_func(x) for x in x_range]
    mode_indices = find_mode_locations(density_func)
    mode_locations = x_range[mode_indices]
    return mode_locations

# get performance out of predictions
def get_performance(predicted_modes, true_mode):
    """
    Gets performance from array of predicted modes.

    Parameters:
    predicted_modes (list or numpy.ndarray): A list or array of numeric values.
    true_mode : one value

    Returns:
    scalar: A performance value.
    """
    pred_arr = np.array(predicted_modes)
    return np.min(np.abs(pred_arr-true_mode))

# normalise a distribution
def normalise_array(arr):
    """
    Normalise

    Parameters:
    arr (list or numpy.ndarray): A list or array of numeric values.

    Returns:
    arr: normalised array
    """
    return np.array(arr)/np.sum(arr)