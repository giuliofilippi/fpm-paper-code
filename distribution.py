# imports
import numpy as np
import matplotlib.pyplot as plt

# get weighted distribution
def get_weighed_distribution(novelty_diff, novelty_sum, n_saccades, roll):
    # we keep sampling saccades until we have n_saccade of them
    # we first sample directions from novelty sum dist, and then for 
    # each direction we look at the novelty difference at that direction. 
    # We accept that direction as a saccade with probability 1-|epsilon| 
    # where epsilon is the novelty difference value.
    saccades = []
    directions = np.arange(0, 360)

    # useful variables
    max_diff = np.max(np.abs(novelty_diff))
    min_sum = np.min(novelty_sum)
    max_sum_delta = np.max(novelty_sum) - min_sum
    
    while len(saccades) < n_saccades:
        # Sample a direction uniformly at random
        direction = np.random.choice(directions)
        
        # Evaluate the novelty difference at the sampled direction
        sum_delta = novelty_sum[direction] - min_sum
        diff_delta = novelty_diff[direction]
        
        # Compute the probability of acceptance
        prob_sum = np.e**(-4*(sum_delta/max_sum_delta))
        prob_diff = np.e**(-4*np.abs(diff_delta)/max_diff)
        
        # Accept the direction as a saccade with probability prob_sum * prob_diff
        if np.random.rand() < prob_sum * prob_diff:
            saccades.append(direction-180+roll)
    
    # return
    return np.array(saccades)

# returns the highest mode location
def find_highest_mode_location(dist):
    # Calculate the histogram
    counts, bin_edges = np.histogram(dist, bins=int((np.max(dist) - np.min(dist)) / 5))

    # Find the index of the bin with the highest count
    max_bin_index = np.argmax(counts)

    # Find the location of the highest bin (mode)
    mode_location = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    # return
    return mode_location

# plot histogram function (no longer used in results)
def plot_histogram(dist):
    # create figure
    _, ax = plt.subplots(figsize=(6.4, 6.4 / 2))

    # Plot histogram with density=True to normalize it
    ax.hist(dist, bins=int((np.max(dist)-np.min(dist))/5), 
                               color='black', edgecolor='dimgrey', linewidth=0.2, density=True)
    
    
    # Format the y-axis to display percentage
    #ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # return the axis object
    return ax