import numpy as np
# import skimage as sk
import skimage.io as skio
# from skimage.transform import rescale
# from scipy import ndimage
import random
# import scipy.sparse
import scipy.sparse.linalg


# Function that takes in an image, a pixel within the image, and the image width.
# and returns the index of the pixel in the flattened image
def get_nbhd(img_shape, pix_coord, nbhd_width):
    row_slc_0 = pix_coord[0] - nbhd_width // 2 if (pix_coord[0] - nbhd_width // 2) >= 0 else 0
    row_slc_1 = pix_coord[0] + nbhd_width // 2 if (pix_coord[0] + nbhd_width // 2) < img_shape[0] else img_shape[0] - 1

    col_slc_0 = pix_coord[1] - nbhd_width // 2 if (pix_coord[1] - nbhd_width // 2) >= 0 else 0
    col_slc_1 = pix_coord[1] + nbhd_width // 2 if (pix_coord[1] + nbhd_width // 2) < img_shape[1] else img_shape[1] - 1

    slices = [np.arange(row_slc_0, row_slc_1 + 1), np.arange(col_slc_0, col_slc_1 + 1)]
    # nbhd_pixels = img[row_slc_0: row_slc_1 + 1, col_slc_0:col_slc_1 + 1].flatten()

    return np.apply_along_axis(lambda x: img_shape[1] * x[0] + x[1], 1,
                               np.array([(i, j) for i in slices[0] for j in slices[1]])).astype(int)


# For all patches in X, find the patch in Z that is the nearest neighbor, and return the mapping
def nearest_neighbors(X, Z):
    pass


# name of the input file
imname = 'texture.jpg'
# read in the image
Z_src = skio.imread(imname)
Z_flat = Z_src.reshape((Z_src[:, :, 0].size, 3))

nbhd_width = 31  # make sure odd, so there can be a center nbhd point, or else stupid annoying bug might occur

out = np.zeros((200, 200, 3), dtype=np.uint8)
out_flat = out.reshape((out[:, :, 0].size, 3))

assert (nbhd_width % 2 == 1)  # needs to be odd or else potential bugs
# Create the set of points that are the centers of the neighborhoods we will be comparing
x_centers = set(np.arange(nbhd_width // 2, out.shape[0] - nbhd_width // 2, nbhd_width // 4))
x_centers.add(out.shape[0] - nbhd_width // 2 - 1)  # make sure to cover every output pixel

y_centers = set(np.arange(nbhd_width // 2, out.shape[1] - nbhd_width // 2, nbhd_width // 4))
y_centers.add(out.shape[1] - nbhd_width // 2 - 1)  # make sure to cover every output pixel

# Cartesian product of the x and y partitions
prod = np.transpose([np.tile(list(x_centers), len(y_centers)), np.repeat(list(y_centers), len(x_centers))])
nbhd_centers = set([tuple(prod[i]) for i in range(len(prod))])

# Map from output patches to input patches that are the nearest neighbors
current_neighbors = dict.fromkeys(nbhd_centers)

# Initialize the nearest neighbors for each nbhd in X(output) to random nbhds in Z(source)
for patch in current_neighbors:
    # x_rand = random.randrange(0, Z_src.shape[0])
    # y_rand = random.randrange(0, Z_src.shape[1])
    randrow = random.randrange(nbhd_width // 2, Z_src.shape[0] - nbhd_width // 2)
    randcol = random.randrange(nbhd_width // 2, Z_src.shape[1] - nbhd_width // 2)
    current_neighbors[patch] = (randrow, randcol)

N = 1  # number of iterations
for i in range(N):
    diags = np.zeros(out_flat.shape)  # just the diagonals of the matrix A
    b = np.zeros(out_flat.shape)  # vector b on the RHS of matrix equation Ax = b

    # For each output patch, loop through each pixel in the input patch and NN-output patch, and increment the entry
    # along the diagonal of matrix A corresponding to the output patch pixel's index in the full size output image
    # vector. At the same time, for the corresponding nearest neighbor output patch pixel, put that pixel value in the
    # same index in the b vector.
    for patch in current_neighbors:
        # indices contains a list of integers that are the diagonals to update in the matrix A
        # aka indices of neighborhood pixels in the flattened image

        indices = get_nbhd(out.shape, patch, nbhd_width)
        z_ind = get_nbhd(Z_src.shape, current_neighbors[patch], nbhd_width)

        z_pixels = Z_flat[z_ind]
        # update the b vector
        # increment the diagonals of A
        diags[indices] += 1
        try:
            b[indices] += z_pixels
        except:
            print("error:")
            print("current_patch: {0}", patch)
            print("out patch shape: {0}", indices.shape)
            print("source patch shape: {0}", z_ind.shape)
            exit()

    A = scipy.sparse.spdiags(diags.flatten('F'), 0, 3*out_flat.shape[0], 3*out_flat.shape[0], 'csr')
    flattened_sol = scipy.sparse.linalg.spsolve(A, b.flatten('F'))
    updated_image = flattened_sol.reshape(b.shape, order='F').reshape(out.shape).astype(np.uint8)

    # Follow rest of algorithm (1) in the paper:
    # Compute the nearest neighbors of updated_image
    # Maybe use some efficient library for the nearest neighbors computation?
    # **NOTE** Remember that LazyFluids reverses the direction of the NNF retreival:
    # That is, for each source patch, z_p we find a target patch x_q that has minimal distance.
    updated_neighbors = nearest_neighbors(updated_image, Z_src)

    out = updated_image
    # If updated neighbors is the same as current neighbors, then exit the loop and set the output image to the
    # updated image

# save the image
# fname = 'output.jpg'
# skio.imsave(fname, im)

# display the image
skio.imshow(out)
skio.show()
