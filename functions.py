# imports
import numpy as np

# MSE function between feature vectors (vectorized)
def MSE(v1, v2):
    """
    Compute the mean squared error between two feature vectors.

    Parameters:
    v1 (numpy.ndarray): Tensor with feature vectors along the last axis.
    v2 (numpy.ndarray): Tensor with feature vectors along the last axis.
    """
    return np.mean((v1-v2)**2, axis=-1)

# Image Difference function between images (vectorized)
def IDF(im1, im2):
    """
    Compute the mean squared error between two matrices.

    Parameters:
    im1 (numpy.ndarray): New matrix. Tensor with images along the last two axes.
    im2 (numpy.ndarray): Stored matrix. Tensor with images along the last two axes.
    """

    if im1.shape[-2:]==im2.shape[-2:]:
        return np.mean((im1 - im2) ** 2, axis=(-2,-1))
        
    else:
        raise ValueError("Matrices should match along last two dims")

# Correlation function between images (vectorized)
def CORR(im1, im2):
    """
    Compute the correlation between two matrices.

    Parameters:
    im1 (numpy.ndarray): New matrix. Tensor with images along the last two axes.
    im2 (numpy.ndarray): Stored matrix. Tensor with images along the last two axes.
    """
    if im1.shape[-2:]==im2.shape[-2:]:
        # Compute mean along the last axes
        mean_v1 = np.mean(im1, axis=(-2,-1), keepdims=True)
        mean_v2 = np.mean(im2, axis=(-2,-1), keepdims=True)

        # Compute covariance along the last axes
        cov = np.mean((im1 - mean_v1) * (im2 - mean_v2), axis=(-2,-1))

        # Compute standard deviation along the last axes
        std_v1 = np.std(im1, axis=(-2,-1))
        std_v2 = np.std(im2, axis=(-2,-1))

        # Compute correlation along the last axes
        correlation = cov / (std_v1 * std_v2)

        return correlation
    
    else:
        raise ValueError("Matrices should match along last two dims")

# jaccard index of two matrices
def JACCARD(im1, im2):
    """
    Compute the Jaccard index between two matrices.

    Parameters:
    im1 (numpy.ndarray): New matrix. Tensor with images along the last two axes.
    im2 (numpy.ndarray): Stored matrix. Tensor with images along the last two axes.
    """
    if im1.shape[-2:] == im2.shape[-2:]:
        # Compute the intersection and union
        intersection = np.sum(np.logical_and(im1 == 1, im2 == 1), axis=(-2, -1))
        union = np.sum(np.logical_or(im1 == 1, im2 == 1), axis=(-2, -1))
        
        # Compute the Jaccard index
        jaccard_index = intersection / union

        return jaccard_index
    
    else:
        raise ValueError("Matrices should match along last two dims")

# get rotations of an image (360 pixel assumption)
def get_rotations(im):
    """
    Generate all 360 rotations of an image, assuming a 360 pixel width.

    Parameters:
    im (numpy.ndarray): Input image. Assumes the image has a width of 360 pixels.

    Returns:
    numpy.ndarray: A tensor containing 360 rotated versions of the input image.
    """
    rotations = [im]
    prev_im = im
    for i in range(359):
        new_im = np.roll(prev_im, shift=-1, axis=1)
        rotations.append(new_im)
        prev_im = new_im
    return np.array(rotations)

# get shifted rotations of an image (360 pixel assumption)
def get_shifted_rotations(im):
    """
    Generate all 360 rotations of an image, assuming a 360 pixel width.

    Parameters:
    im (numpy.ndarray): Input image. Assumes the image has a width of 360 pixels.

    Returns:
    numpy.ndarray: A tensor containing 360 rotated versions of the input image.
    """
    rotations = [np.roll(im, shift = 180, axis=1)]
    prev_im = np.roll(im, shift = 180, axis=1)
    for i in range(359):
        new_im = np.roll(prev_im, shift=-1, axis=1)
        rotations.append(new_im)
        prev_im = new_im
    return np.array(rotations)

# compute fpm function
def compute_fpm(image):
    """
    Computes FPM of image.

    Parameters:
    image (numpy.ndarray): The image

    Returns:
    FPM value
    """
    return np.sum(image[:,:180])/np.sum(image)

# compute sfpm
def compute_sfpm(image):
    """
    Computes sFPM of image.

    Parameters:
    image (numpy.ndarray): The image

    Returns:
    sFPM value
    """
    # find segmentation point
    for x in range(150,210):
        if np.sum(image[:,x])<=1:
            seg = x
    return np.sum(image[:,:180])/np.sum(image[:,:seg])

# theta angle from FPM function for given shape
def x_from_fpm(image, fpm):
    """
    Get angle at which integral equals FPM.

    Parameters:
    image (numpy.ndarray): The image
    fpm (float): The FPM value

    Returns:
    theta (angle)
    """
    _, width = image.shape
    S = 0
    T = np.sum(image)
    for k in range(width):
        S += np.sum(image[:,k])
        if S/T > fpm:
            return k