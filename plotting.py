# imports
import numpy as np
import matplotlib.pyplot as plt
from statistical import gaussian_kde, normalise_array

# plot an image with optional FPM location setting
def plot_image(image, title='Image', fpm=None, plf=None):
    """
    Plot an image with optional vertical lines.

    Parameters:
    image (numpy.ndarray): The image to be plotted.
    title (str): The title of the plot. Default is 'Image'.
    fpm (float, optional): The x-coordinate for an additional vertical line to be plotted in green. Default is None.

    Returns:
    None
    """
    # Plot the image
    plt.figure(figsize=(6.4, 6.4/2))
    plt.imshow(image, cmap='binary', extent=(0, image.shape[1], 0, image.shape[0]))
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axvline(x=image.shape[1]/2, color='red', linestyle='--', label='Visual Center')
    if fpm is not None:
        plt.axvline(x=fpm, color='green', linestyle='--', label='FPM match')
    if plf is not None:
        plt.axvline(x=plf, color='blue', linestyle='--', label='pLF match')
    plt.legend()
    ax = plt.gca()
    return ax

# plot an image with optional FPM location setting
def plot_rolled_image(image, title='Image', fpm=None, plf=None, roll=None, feeder=None):
    """
    Plot an image with optional vertical lines.

    Parameters:
    image (numpy.ndarray): The image to be plotted.
    title (str): The title of the plot. Default is 'Image'.
    fpm (float, optional): The x-coordinate for an additional vertical line to be plotted in green. Default is None.

    Returns:
    None
    """
    # Plot the image
    plt.figure(figsize=(6.4, 6.4/2))
    plt.imshow(np.roll(image,-180+roll), cmap='binary', extent=(0, image.shape[1], 0, image.shape[0]))
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    if feeder is not None:
        plt.axvline(x=roll, color='white', linestyle='-', label='Feeder')
    if fpm is not None:
        plt.axvline(x=fpm, color='green', linestyle='--', label='FPM match')
    if plf is not None:
        plt.axvline(x=plf, color='blue', linestyle='--', label='pLF match')
    ax = plt.gca()
    return ax

# plot multiple images
def plot_images(image_list, title='Images'):
    """
    Plot images.

    Parameters:
    image_list (numpy.ndarray): The image list to be plotted.
    title (str): The title of the plot. Default is 'Images'.

    Returns:
    None
    """
    # Create a new figure with adjusted size
    num_images = len(image_list)
    fig, axs = plt.subplots(num_images, figsize=(4, 2*num_images))
    fig.suptitle(title)
    # Generate and plot the images
    for i,im in enumerate(image_list):        
        # Plot the image
        plt.subplot(num_images, 1, i+1)
        plt.imshow(im, cmap='binary', extent=(0, im.shape[1], 0, im.shape[0]))
        plt.axvline(x=im.shape[1]/2, color='red', linestyle='--', label='Visual Center')

    plt.tight_layout()
    plt.show()

# plot image special with labels
def plot_image_s(image, title='Image', sfpm=None, wfpm=None, elf=None):
    # Plot the image
    plt.figure(figsize=(6.4, 6.4/2))
    plt.imshow(image, cmap='binary', extent=(0, image.shape[1], 0, image.shape[0]))
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axvline(x=image.shape[1]/2, color='red', linestyle='--', label='Visual Center')
    if sfpm is not None:
        plt.axvline(x=sfpm, color='green', linestyle='--', label='sFPM')
    if wfpm is not None:
        plt.axvline(x=wfpm, color='blue', linestyle='--', label='wFPM')
    if elf is not None:
        plt.axvline(x=elf, color='blue', linestyle='--', label='eLF')
    plt.legend()
    plt.show()

# plot distribution image
def plot_distribution_image(dist_pool, title='Distribution', y_lim=0.2):
    # Plotting
    fig = plt.figure(figsize=(6.4, 6.4 / 2))
    bins = np.arange(-65, 210, 5) # bins for distributions

    # Plot histogram with density=True to normalize it
    counts, _ = np.histogram(dist_pool, bins=bins)
    probabilities = counts / np.sum(counts)  # Normalize counts to sum to 1
    plt.bar(bins[:-1], probabilities, width=5, edgecolor='dimgrey', linewidth=0.2, color='black', align='center')
    plt.title('Distribution')
    plt.xlim(-67, 200)
    plt.xlabel('pattern width (°)')
    plt.ylabel('saccade percentage (%)')

    # Plot KDE
    x_range = np.arange(-70, 205, 1)
    kde_func = gaussian_kde(dist_pool, h=5)
    density_func = [kde_func(x) for x in x_range]
    plt.plot(x_range, normalise_array(density_func)*5, label='Gaussian KDE')

    # Formatting
    ax = plt.gca()
    plt.tight_layout()
    ax.set_position([0.1, 0.1, 0.9, 0.8])
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Define ticks from -50 to 200 in steps of 25
    ticks = np.arange(-50, 225, 25)
    ax.set_xticks(ticks)
    # Label only every other tick
    labels = [str(tick) if i % 2 == 0 else '' for i, tick in enumerate(ticks)]
    ax.set_xticklabels(labels)
    # Define y ticks from 0 to y_lim in steps of 0.005
    yticks = np.arange(0, y_lim+0.05, 0.05)  # Include 0.05
    ax.set_yticks(yticks)
    # Label only every other y tick (i.e., every 0.01)
    ylabel = [f'{tick:.1f}' if i % 2 == 0 else '' for i, tick in enumerate(yticks)]
    ax.set_yticklabels(ylabel)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 100:.0f}'))