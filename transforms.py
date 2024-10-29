# imports
import numpy as np
from scipy import ndimage

# downsample ar by a factor of fact
def block_mean(ar, fact):
    # block mean code
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res

# downsampled feature transform
def downsample_transform(arr, fact=4):
    # crop and downsample
    arr_cropped = arr[10:,:]
    downsampled_image = block_mean(arr_cropped, fact)

    # returns (20, 90) image
    return downsampled_image

# go from (90, 360) image to eye model feature vector
def eye_model_transform(array, left_matrix, right_matrix):
    # Split the last dimension into two parts of size 180
    left_view, right_view = np.split(array, 2, axis=1)

    # Sample positions using the binary matrices
    left_sampled = left_view[left_matrix.astype(bool)]
    right_sampled = right_view[right_matrix.astype(bool)]

    # feature vector
    feat_vec = np.concatenate((left_sampled, right_sampled))

    # return (20, 90) shape vector
    return feat_vec

# flatten transform
def flatten_transform(arr):
    half = int(arr.shape[1]/2)
    # split
    left = arr[:,:half].flatten()
    right = arr[:,half:].flatten()
    
    # returns flat image
    return np.concatenate((left, right))

# downsample and flatten transform
def downsample_and_flatten_transform(arr, fact=4):
    # downsample and flatten
    return flatten_transform(downsample_transform(arr, fact))

# just overlap transform
def overlap_transform(arr, overlap=10):
    # half overlap
    delta = int(overlap/2)

    # split
    left_view = arr[:,delta:180+delta]
    right_view = arr[:,180-delta:360-delta]

    # merged view
    merged_view = np.concatenate((left_view, right_view), axis=1)

    # return
    return merged_view

# transform with overlap
def overlap_downsample_transform(arr, overlap=20):
    # half overlap
    delta = int(overlap/2)

    # split
    left_view = arr[:,delta:180+delta]
    right_view = arr[:,180-delta:360-delta]

    # merged view
    merged_view = np.concatenate((left_view, right_view), axis=1)
    downsample_view = downsample_transform(merged_view, fact=4)

    # return
    return flatten_transform(downsample_view)

# go from (90, 360) image to eye model feature vector
def overlap_eye_model_transform(array, left_matrix, right_matrix, overlap):
    # Split the last dimension into two parts of size 180
    left_view, right_view = np.split(overlap_transform(array, overlap), 2, axis=1)

    # Sample positions using the binary matrices
    left_sampled = left_view[left_matrix.astype(bool)]
    right_sampled = right_view[right_matrix.astype(bool)]

    # feature vector
    feat_vec = np.concatenate((left_sampled, right_sampled))

    # return (20, 90) shape vector
    return feat_vec

# mid resolution transform
def mid_res_transform(arr):
    # returns (40, 80) image
    xarr = downsample_transform(arr, fact=2)
    # halfway
    half = int(xarr.shape[1]/2)
    # split
    left = arr[:,:half].flatten()
    right = arr[:,half:].flatten()
    
    # returns flat image
    return np.concatenate((left, right))

# full res transform
def full_res_transform(arr):
    half = int(arr.shape[1]/2)
    # split
    left = arr[:,:half].flatten()
    right = arr[:,half:].flatten()
    
    # returns flat image
    return np.concatenate((left, right))