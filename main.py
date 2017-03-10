import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from scipy import ndimage
import random
import scipy.sparse
import scipy.sparse.linalg


# Function that takes in an image, a pixel within the image, and a nbhd width. Returns an array containing the nbhd of
# pixels
def get_nbhd(img, pix_coord, width):
    # pixel_index = patch[0] * out.shape[1] + patch[1]
    pass


# For all patches in X, find the patch in Z that is the nearest neighbor, and return the mapping
def nearest_neighbors(X, Z):
    pass

# name of the input file
imname = 'texture.jpg'
# read in the image
Z_src = skio.imread(imname)
nbhd_width = 32
# output image X, size is 800 by 800
out = np.ndarray((400, 400, 3))

# Create the set of points that are the centers of the neighborhoods we will be comparing
x_centers = np.arange(nbhd_width / 2, out.shape[0] - nbhd_width / 2, nbhd_width / 4)
y_centers = np.arange(nbhd_width / 2, out.shape[1] - nbhd_width / 2, nbhd_width / 4)

# Cartesian product of the x and y partitions
prod = np.transpose([np.tile(x_centers, len(y_centers)), np.repeat(y_centers, len(x_centers))])
nbhd_centers = set([tuple(prod[i]) for i in range(len(prod))])

# Map from output patches to input patches that are the nearest neighbors
current_neighbors = dict.fromkeys(nbhd_centers)

# Initialize the nearest neighbors for each nbhd in X(output) to random nbhds in Z(source)
for patch in current_neighbors:
    # x_rand = random.randrange(0, Z_src.shape[0])
    # y_rand = random.randrange(0, Z_src.shape[1])
    x_rand = random.randrange(nbhd_width / 2, Z_src.shape[0] - nbhd_width / 2 + 1)
    y_rand = random.randrange(nbhd_width / 2, Z_src.shape[1] - nbhd_width / 2 + 1)
    current_neighbors[patch] = (x_rand, y_rand)

N = 1  # number of iterations
for i in range(N):
    diags = np.zeros(out[:, :, 0].size)  # just the diagonals of the matrix A
    b = np.zeros(out[:, :, 0].size)  # vector b on the RHS of matrix equation Ax = b

    # For each output patch, loop through each pixel in the input patch and NN-output patch, and increment the entry
    # along the diagonal of matrix A corresponding to the output patch pixel's index in the full size output image
    # vector. At the same time, for the corresponding nearest neighbor output patch pixel, put that pixel value in the
    # same index in the b vector.
    for patch in current_neighbors:
        # indices contains a list of integers that are the diagonals to update in the matrix A
        # aka indices of neighborhood pixels in the flattened image
        indices = get_nbhd(out, patch, out.shape[1])
        z_pixels = Z_src[:, :, 0].flatten()[get_nbhd(Z_src, current_neighbors[patch], Z_src.shape[1])]

        # update the b vector
        b[indices] += z_pixels
        # increment the diagonals of A
        diags[indices] += 1

    updated_image = scipy.sparse.linalg.solve_triangular(scipy.sparse.diags(diags), b)

    # Follow rest of algorithm (1) in the paper:
    # Compute the nearest neighbors of updated_image
    # Maybe use some efficient library for the nearest neighbors computation?
    # **NOTE** Remember that LazyFluids reverses the direction of the NNF retreival:
    # That is, for each source patch, z_p we find a target patch x_q that has minimal distance.
    updated_neighbors = nearest_neighbors(updated_image, Z_src)
    # If updated neighbors is the same as current neighbors, then exit the loop and set the output image to the
    # updated image

# save the image
# fname = 'output.jpg'
# skio.imsave(fname, im)

# display the image
skio.imshow(Z_src)
skio.show()
