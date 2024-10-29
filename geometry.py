# imports
import numpy as np
from plotting import plot_image, plot_images

# variables
R = 1.5
Z = np.array([R, 0])
pw_angles = np.arange(-180,180)
theta_angles = np.array([-np.deg2rad(pw) for pw in pw_angles])
plot_angles = np.arange(0,360)
point_list = [R*np.array([np.cos(theta), np.sin(theta)]) for theta in theta_angles]

# transform angle
def pw_to_theta(pw):
    """
    Transforms coordingates
    
    Parameters:
    pw (degrees): The angle in degrees
    
    Returns:
    float: The angle in radians
    """
    return np.deg2rad(-pw)

# transform angle back
def theta_to_pw(theta):
    """
    Transforms coordingates
    
    Parameters:
    pw (degrees): The angle in degrees
    
    Returns:
    float: The angle in radians
    """
    return -np.rad2deg(theta)

# get coordinates
def get_coord(r, pw):
    """
    Get point coordinates
    
    Parameters:
    r (number): radius
    pw (degrees): The angle in degrees
    
    Returns:
    arr: The point
    """
    theta = pw_to_theta(pw)
    point = r*np.array([np.cos(theta), np.sin(theta)])
    return point

# vector between points
def get_vec(point1, point2):
    """
    Calculates the vector from point1 to point2.
    
    Parameters:
    point1 (tuple): The first point as a tuple (x1, y1).
    point2 (tuple): The second point as a tuple (x2, y2).
    
    Returns:
    float: The vector between the two points.
    """
    return np.array(point2) - np.array(point1)

# distance function
def distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.
    
    Parameters:
    point1 (tuple): The first point as a tuple (x1, y1).
    point2 (tuple): The second point as a tuple (x2, y2).
    
    Returns:
    float: The Euclidean distance between the two points.
    """
    # Convert points to numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Calculate the distance
    distance = np.linalg.norm(point2 - point1)
    
    return distance

# augment the original vertex set
def augment_vertex_set(vertices):
    """
    Augments vertex set with more vertices. DIFFERENT
    THETA CONVENTION 0 ----> 360 (IMPORTANT)
    
    Parameters:
    vertices (list of tuples): List of vertices, each containing (theta, phi).
    
    Returns:
    numpy.ndarray: Array of vertices
    """
    # Convert vertices to numpy array for efficient computation
    vertices = np.array(vertices)
    thetas = vertices[:, 0]
    phis = vertices[:, 1]

    # Prepare array to hold new vertices
    new_vertices = []

    # Iterate over all angles theta from 0 to 359
    for theta in range(360):
        # Find the indices of the two nearest vertices for interpolation
        idx2 = np.searchsorted(thetas, theta, side='right')
        idx1 = idx2 - 1 if idx2 > 0 else len(thetas) - 1
        idx2 = idx2 if idx2 < len(thetas) else 0
        
        # Get the thetas and phis for interpolation
        theta1, phi1 = thetas[idx1], phis[idx1]
        theta2, phi2 = thetas[idx2], phis[idx2]

        # Adjust for the circular nature of the angles
        if theta1 > theta2:
            theta2 += 360
        if theta < theta1:
            theta += 360

        # Perform linear interpolation
        phi_interpolated = phi1 + (phi2 - phi1) * (theta - theta1) / (theta2 - theta1)

        # Append the new vertex
        new_vertices.append([theta % 360, phi_interpolated])

    return np.array(new_vertices)

# get polar angle
def compute_phi(d, phi):
    """
    Calculates the Polar ange phi from distance and height.
    
    Parameters:
    d (distance): distance
    z (height): height
    
    Returns:
    float: phi.
    """
    z = R * np.tan(np.deg2rad(phi))
    return np.rad2deg(np.arctan(z/d))

# map azimuths
def azimuth_map(L):
    """
    Azimuth map.
    """
    azimuth_list = []
    distance_list = []

    for i,P in enumerate(point_list):
        # distances
        LZ = distance(L, Z)
        LP = distance(L, P)
        # dot product
        dot_product = np.dot(Z - L, P - L)
        theta_angle = np.arccos(np.clip(dot_product/(LZ * LP), -1, 1))
        pw_angle = theta_to_pw(theta_angle)
        # make sure you get right sign
        if i <= 180:
            sign = 1
        else:
            sign = -1
        # add to list
        azimuth_list.append(sign*pw_angle)
        distance_list.append(LP)

    return azimuth_list, distance_list

# draw a shape on a 90 by 360 canvas using height array
def draw_shape(heights, im_height=90, im_width=360):
    """
    Draws the shape given a list of heights of lenght 360.
    Each height is the polar angle.
    
    Parameters:
    heights (list) : heights list
    """
    # initialize image
    image = np.zeros((im_height, im_width))
    for i in range(im_width):
        height = int(heights[i]) # integer
        image[im_height-height:,i] = 1
    # return image
    return image

# draw image
def draw_image(vertices):
    # interpolate the heights of mapped vertices
    heights = augment_vertex_set(vertices)[:,1]
    # draw the shape given by these heights
    shape = draw_shape(heights, im_height=90, im_width=360)
    # append shape to train images
    return shape

# function to get all rotations at a particular location L
def map_vertices(augmented_vertices, L, azimuth_list, distance_list):
    """
    Map vertices to view from other location L.
    """
    # new vertices
    new_vertices = np.zeros_like(augmented_vertices)
    new_thetas = np.array(azimuth_list)+180
    new_phis = [compute_phi(distance_list[i], augmented_vertices[i, 1]) for i in range(360)]
    new_vertices[:, 0] = new_thetas
    new_vertices[:, 1] = new_phis
    # return
    return new_vertices

# function to get singal rotation at a location L
def get_single_rotations(vertices, L):
    """
    Get single rotation at location L.
    """
    # augment vertices
    augmented_vertices = augment_vertex_set(vertices)
    azimuth_list, distance_list = azimuth_map(L)
    mapped_vertices = map_vertices(augmented_vertices, L, azimuth_list, distance_list)
    augmented_mapped_vertices = augment_vertex_set(mapped_vertices)
    new_heights = augmented_mapped_vertices[:,1]
    base_shape = draw_shape(new_heights)

    # return
    return base_shape

# function to get all rotations at a location L
def get_all_rotations(vertices, L):
    """
    Get all rotations at location L.
    """
    # init list
    rotations_list = []
    # augment vertices
    augmented_vertices = augment_vertex_set(vertices)
    azimuth_list, distance_list = azimuth_map(L)
    mapped_vertices = map_vertices(augmented_vertices, L, azimuth_list, distance_list)
    augmented_mapped_vertices = augment_vertex_set(mapped_vertices)
    new_heights = augmented_mapped_vertices[:,1]
    base_shape = draw_shape(new_heights)
    # for each possible roll add shape
    for az in azimuth_list:
        int_az = int(az)
        rot_im = np.roll(base_shape, shift=-int_az, axis=1)
        rotations_list.append(rot_im)
    # return
    rotations = np.array(rotations_list)
    return rotations