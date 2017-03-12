import numpy as np
# import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from scipy import ndimage
import random
# import scipy.sparse
import scipy.sparse.linalg
from sklearn.neighbors import NearestNeighbors

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


# name of the input file
imname = 'texture.jpg'
# read in the image
Z_src = skio.imread(imname)
Z_src = np.round(rescale(Z_src, 100 / Z_src.shape[0], preserve_range=True)).astype(np.uint8)

Z_alpha = 255 - skio.imread("texture_alpha.jpg", as_grey=True).astype(np.uint8)
Z_alpha = Z_alpha.reshape((Z_alpha.shape[0], Z_alpha.shape[1], 1))
Z_src = np.append(Z_src, Z_alpha, 2)

Z_flat = Z_src.reshape((Z_src[:, :, 0].size, 4))

nbhd_width = 31  # make sure odd, so there can be a center nbhd point, or else stupid annoying bug might occur

out = np.zeros((200, 200, 3), dtype=np.uint8)
out_alpha = 255 - skio.imread("target_alpha.jpg", as_grey=True).astype(np.uint8)
out_alpha = out_alpha.reshape((out_alpha.shape[0], out_alpha.shape[1], 1))
out = np.append(out, out_alpha, 2)

out_flat = out.reshape((out[:, :, 0].size, 4))


# Custom distance metric where x and y are two "points" in patch space. i.e. R^n where n = total pixels in nbhd
def dist_metric(x, y):
    # x_px = out_flat[get_nbhd(out.shape, x, nbhd_width)]
    # y_px = out_flat[get_nbhd(out.shape, y, nbhd_width)]
    alpha_blur_x, alpha_blur_y = 255, 255
    eta = 0
    squared_diff = (x - y) ** 2
    return 3 * np.sum(squared_diff[nbhd_width ** 2 * 3:]) + np.sum(squared_diff[:nbhd_width ** 2 * 3]) + eta * ((alpha_blur_x - alpha_blur_y) ** 2)


assert (nbhd_width % 2 == 1)  # needs to be odd or else potential bugs
# Create the set of points that are the centers of the neighborhoods we will be comparing
x_centers = set(np.arange(nbhd_width // 2, out.shape[0] - nbhd_width // 2, nbhd_width // 4))
x_centers.add(out.shape[0] - nbhd_width // 2 - 1)  # make sure to cover every output pixel

y_centers = set(np.arange(nbhd_width // 2, out.shape[1] - nbhd_width // 2, nbhd_width // 4))
y_centers.add(out.shape[1] - nbhd_width // 2 - 1)  # make sure to cover every output pixel

# Cartesian product of the x and y partitions
prod = np.transpose([np.tile(list(x_centers), len(y_centers)), np.repeat(list(y_centers), len(x_centers))])
X_nbhd_centers = [tuple(prod[i]) for i in range(len(prod))]

# Create the set of points that are the centers of the neighborhoods for the Source Image
x_centers = set(np.arange(nbhd_width // 2, Z_src.shape[0] - nbhd_width // 2, nbhd_width // 4))
x_centers.add(Z_src.shape[0] - nbhd_width // 2 - 1)  # make sure to cover every pixel

y_centers = set(np.arange(nbhd_width // 2, Z_src.shape[1] - nbhd_width // 2, nbhd_width // 4))
y_centers.add(Z_src.shape[1] - nbhd_width // 2 - 1)  # make sure to cover every pixel

# Cartesian product of the x and y partitions
prod = np.transpose([np.tile(list(x_centers), len(y_centers)), np.repeat(list(y_centers), len(x_centers))])
Z_nbhd_centers = [tuple(prod[i]) for i in range(len(prod))]

# Map from output patches to input patches that are the nearest neighbors
current_neighbors = dict.fromkeys(X_nbhd_centers)

# Initialize the nearest neighbors for each nbhd in X(output) to random nbhds in Z(source)
# nbhds = []  # ugh its 7am and I'm too lazy to use some numpy way to make this cleaner, pls just work
for patch in X_nbhd_centers:
    # x_rand = random.randrange(0, Z_src.shape[0])
    # y_rand = random.randrange(0, Z_src.shape[1])
    current_neighbors[patch] = random.sample(Z_nbhd_centers, 1)[0]
    # nbhds.append(get_nbhd(out.shape, patch, nbhd_width))
# nbhds = out_flat[np.array(nbhds)]
# get_nbhd(out.shape,list(X_nbhd_centers)[0],nbhd_width)

NNF = np.ndarray((len(X_nbhd_centers), nbhd_width ** 2, 4), dtype=np.uint8)
N = 1  # number of iterations
for i in range(N):
    diags = np.zeros(out_flat.shape)  # just the diagonals of the matrix A
    b = np.zeros(out_flat.shape)  # vector b on the RHS of matrix equation Ax = b

    # For each output patch, loop through each pixel in the input patch and NN-output patch, and increment the entry
    # along the diagonal of matrix A corresponding to the output patch pixel's index in the full size output image
    # vector. At the same time, for the corresponding nearest neighbor output patch pixel, put that pixel value in the
    # same index in the b vector.
    for patch in X_nbhd_centers:
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

    A = scipy.sparse.spdiags(diags.flatten('F'), 0, 4 * out_flat.shape[0], 4 * out_flat.shape[0], 'csr')
    flattened_sol = scipy.sparse.linalg.spsolve(A, b.flatten('F'))
    updated_image = flattened_sol.reshape(b.shape, order='F').reshape(out.shape).astype(np.uint8)

    # Follow rest of algorithm (1) in the paper:
    # Compute the nearest neighbors of updated_image
    # Maybe use some efficient library for the nearest neighbors computation?
    # **NOTE** Remember that LazyFluids reverses the direction of the NNF retreival:
    # That is, for each source patch, z_p we find a target patch x_q that has minimal distance.
    for k in range(len(X_nbhd_centers)):
        patch = X_nbhd_centers[k]
        NNF[k] = updated_image.reshape((updated_image[:, :, 0].size, 4))[get_nbhd(updated_image.shape, patch, nbhd_width)]

    NNF = NNF.reshape((NNF.shape[0], NNF.shape[1] * NNF.shape[2]), order='F')
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric=dist_metric).fit(NNF)
    distances, indices = nbrs.kneighbors()

    test = Z_flat[get_nbhd(Z_src.shape, Z_nbhd_centers[0], nbhd_width)].flatten('F')
    # updated_neighbors = nearest_neighbors(updated_image, Z_src)

    out = updated_image
    # If updated neighbors is the same as current neighbors, then exit the loop and set the output image to the
    # updated image

# save the image
# fname = 'output.jpg'
# skio.imsave(fname, im)

# display the image
skio.imshow(out)
skio.show()
