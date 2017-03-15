import boundaries
import numpy as np
import scipy.sparse.linalg
import skimage.io as skio
from skimage.transform import rescale
from sklearn.neighbors import NearestNeighbors
from skimage.transform import warp
from skimage.transform import SimilarityTransform

class Patch:
    def __init__(self, is_Z, img_shape, center, width=31):
        self.is_src = is_Z
        self.im_shape = img_shape
        self.center_pix_loc = center
        self.width = 31

        if self.is_src:
            self.nn_patch = list()  # list of patches if src patch
        else:
            self.nn_patch = None

        self.pixel_indices = None
        self.set_nbhd()

    def set_nbhd(self):
        row_slc_0 = self.center_pix_loc[0] - self.width // 2 if (self.center_pix_loc[0] - self.width // 2) >= 0 else 0
        row_slc_1 = self.center_pix_loc[0] + self.width // 2 if (self.center_pix_loc[0] + self.width // 2) < \
                                                                self.im_shape[0] else self.im_shape[0] - 1

        col_slc_0 = self.center_pix_loc[1] - self.width // 2 if (self.center_pix_loc[1] - self.width // 2) >= 0 else 0
        col_slc_1 = self.center_pix_loc[1] + self.width // 2 if (self.center_pix_loc[1] + self.width // 2) < \
                                                                self.im_shape[1] else self.im_shape[1] - 1

        slices = [np.arange(row_slc_0, row_slc_1 + 1), np.arange(col_slc_0, col_slc_1 + 1)]
        # nbhd_pixels = img[row_slc_0: row_slc_1 + 1, col_slc_0:col_slc_1 + 1].flatten()

        self.pixel_indices = np.apply_along_axis(lambda x: self.im_shape[1] * x[0] + x[1], 1,
                                                 np.array([(i, j) for i in slices[0] for j in slices[1]])).astype(int)

    def set_nn(self, other_patch):
        assert (other_patch.is_src != self.is_src)
        if self.is_src:
            self.nn_patch.append(other_patch)
            other_patch.nn_patch = self
        else:
            self.nn_patch = other_patch
            other_patch.nn_patch.append(self)


def construct_nbhds(is_src, im_shape, nbhd_w):
    x_centers = set(np.arange(nbhd_w // 2, im_shape[0] - nbhd_w // 2, nbhd_w // 4))
    x_centers.add(im_shape[0] - nbhd_w // 2 - 1)  # make sure to cover every output pixel

    y_centers = set(np.arange(nbhd_w // 2, im_shape[1] - nbhd_w // 2, nbhd_w // 4))
    y_centers.add(im_shape[1] - nbhd_w // 2 - 1)  # make sure to cover every output pixel

    # Cartesian product of the x and y partitions
    prod = np.transpose([np.tile(list(x_centers), len(y_centers)), np.repeat(list(y_centers), len(x_centers))])
    return [Patch(is_src, im_shape, coord, nbhd_w) for coord in [tuple(prod[i]) for i in range(len(prod))]]


def main(nbhd_width=31, src_texture='texture.jpg', src_alpha='texture_alpha.jpg', tgt_alpha = 'texture_alpha.jpg'):



    X_prev = np.zeros((100, 100, 4), dtype=np.uint8)

    for frame_num in range(2):
        l = 0.6
        # Custom distance metric where x and y are two "points" in patch space. i.e. R^n where n = total pixels in nbhd
        def dist_metric(x, y):
            eta = 0
            squared_diff = (x - y) ** 2
            x_lambdas = x[nbhd_width ** 2 * 9:]
            y_lambdas = y[nbhd_width ** 2 * 9:]
            l = np.maximum(x_lambdas, y_lambdas)
            l3 = np.append(np.append(l, l), l)
            dist = 3 * np.sum(squared_diff[nbhd_width ** 2 * 3:nbhd_width ** 2 * 4]) \
            + np.sum(squared_diff[:nbhd_width ** 2 * 3])\
            + eta * np.sum(squared_diff[nbhd_width ** 2 * 4:nbhd_width ** 2 * 5]) \
            + np.sum(l3*squared_diff[nbhd_width ** 2 * 5: nbhd_width ** 2 * 8]) \
            + 3 * np.sum(l*squared_diff[nbhd_width ** 2 * 8 : nbhd_width ** 2 * 9])
            return dist

        assert (nbhd_width % 2 == 1)
        # read in the source image aka Z
        Z_src = skio.imread(src_texture)
        Z_src = np.round(rescale(Z_src, 100 / Z_src.shape[0], preserve_range=True)).astype(np.uint8)
        Z_alpha = 255 - skio.imread(src_alpha).astype(np.uint8)[:, :, 0:1]
        # Z_blurred_alpha = np.zeros((Z_src.shape[0], Z_src.shape[1], 1), dtype=np.uint8) + 255
        Z_boundaries = boundaries.separate_boundaries(src_alpha, 5)
        Z_blurred_alpha = 255 - Z_boundaries.blur_alpha.reshape((Z_alpha.shape[0], Z_alpha.shape[1], 1))
        # if the source is an image and not video, temporal rgba will always be the same
        Z_temporal_rgba = np.append(Z_src, Z_alpha, 2)
        # add in 10th channel of "-1s " as the lambda term
        Z_lambda = -1*np.ones((Z_alpha.shape))
        Z_src = np.append(np.append(np.append(np.append(Z_src, Z_alpha, 2), Z_blurred_alpha, 2), Z_temporal_rgba, 2), Z_lambda, 2)
        Z_flat = Z_src.reshape((Z_src[:, :, 0].size, Z_src.shape[2]))

        # Construct output image, aka X
        out_alpha = 255 - skio.imread(tgt_alpha).astype(np.uint8)[:, :, 0:1]
        # out_blurred_alpha = np.zeros((out_alpha.shape[0], out_alpha.shape[0], 1), dtype=np.uint8) + 255
        out_boundaries = boundaries.separate_boundaries(tgt_alpha, 5)
        out_blurred_alpha = 255 - out_boundaries.blur_alpha.reshape((out_alpha.shape[0], out_alpha.shape[1], 1))
        out_temporal_rgba = X_prev
        out = np.zeros((out_alpha.shape[0], out_alpha.shape[1], 3), dtype=np.uint8)
        # 10th channel of lambda*max(0, X_a - X_prev)
        out_lambda = out_alpha - X_prev[:, :, 3:4]
        out_lambda[out_lambda < 0] = 0
        out_lambda = l*(1 - out_lambda)
        out = np.append(np.append(np.append(np.append(out, out_alpha, 2), out_blurred_alpha, 2), out_temporal_rgba, 2), out_lambda, 2)
        out_flat = out.reshape((out[:, :, 0].size, out.shape[2]))

        print(out.shape)
        print(Z_src.shape)

        X_patches = construct_nbhds(False, out.shape, nbhd_width)
        Z_patches = construct_nbhds(True, Z_src.shape, nbhd_width)
        K = len(X_patches) // len(Z_patches)
        # Init nearest neighbors to random source patches
        for patch in X_patches:
            patch.set_nn(np.random.choice(Z_patches, 1)[0])

        # Patches in src will not change, so can pull outside of loop
        Z_patches_for_NN_query = np.ndarray((len(Z_patches), nbhd_width ** 2, Z_src.shape[2]), dtype=np.uint8)
        for k in range(len(Z_patches)):
            Z_patches_for_NN_query[k] = Z_flat[Z_patches[k].pixel_indices]

        Z_patches_for_NN_query = Z_patches_for_NN_query.reshape(
            (Z_patches_for_NN_query.shape[0], Z_patches_for_NN_query.shape[1] * Z_patches_for_NN_query.shape[2]), order='F')

        N = 10  # number of iterations
        for i in range(N):
            print("iteration: {0}".format(i))
            diags = np.zeros((out.shape[0] * out.shape[1], 4))  # just the diagonals of the matrix A
            b = np.zeros((out.shape[0] * out.shape[1], 4))  # vector b on the RHS of matrix equation Ax = b

            # For each output patch, loop through each pixel in the input patch and NN-output patch, and increment the entry
            # along the diagonal of matrix A corresponding to the output patch pixel's index in the full size output image
            # vector. At the same time, for the corresponding nearest neighbor output patch pixel, put that pixel value in the
            # same index in the b vector.
            for x_patch in X_patches:
                z_pixels = Z_flat[x_patch.nn_patch.pixel_indices, 0:4]
                # increment the diagonals of A
                diags[x_patch.pixel_indices] += 1
                # update the b vector
                b[x_patch.pixel_indices] += z_pixels

            A = scipy.sparse.spdiags(diags.flatten('F'), 0, 4 * out.shape[0] * out.shape[1], 4 * out.shape[0] * out.shape[1], 'csr')
            flattened_sol = scipy.sparse.linalg.spsolve(A, b.flatten('F'))
            updated_image = flattened_sol.reshape(b.shape, order='F').reshape((out.shape[0], out.shape[1], 4)).astype(np.uint8)

            # Follow rest of algorithm (1) in the paper:
            # Compute the nearest neighbors of updated_image
            # Maybe use some efficient library for the nearest neighbors computation?
            # **NOTE** Remember that LazyFluids reverses the direction of the NNF retreival:
            # That is, for each source patch, z_p we find a target patch x_q that has minimal distance.

            # Fit a nn field to the new updated image space
            NNF = np.ndarray((len(X_patches), nbhd_width ** 2, out.shape[2]), dtype=np.uint8)  # NN to fit scipy library
            for k in range(len(X_patches)):
                X_patches[k].nn_patch = None
                NNF[k, :, 0:4] = updated_image.reshape((updated_image[:, :, 0].size, 4))[X_patches[k].pixel_indices]
                NNF[k, :, 4:] = out_flat[X_patches[k].pixel_indices, 4:]

            NNF = NNF.reshape((NNF.shape[0], NNF.shape[1] * NNF.shape[2]), order='F')
            nbrs = NearestNeighbors(n_neighbors=NNF.shape[0], algorithm='auto', metric=dist_metric).fit(NNF)

            # Find the nearest neighbors in the updated output image space for each patch Z
            distances, indices = nbrs.kneighbors(Z_patches_for_NN_query)

            for p in Z_patches:
                p.nn_patch = []
            # Basic Patch Counting algorithm. Should improve the results, but need to fix, since we loop through the
            # source patches in the same order every time, instead of resolving collisions by choosing the patch map which
            # has the smallest distance
            R = len(X_patches) % len(Z_patches)
            # while there are still unassigned patches in the output image
            while np.sum([1 if xpatch.nn_patch is None else 0 for xpatch in X_patches]) > 0:
                # Assigning NNs to patches of the source image
                for k in range(len(Z_patches)):
                    if len(Z_patches[k].nn_patch) < K:
                        # the nn query is already sorted by distances
                        for j in range(NNF.shape[0]):
                            # Loop until we find an unused X patch
                            if X_patches[indices[k, j]].nn_patch is None:
                                Z_patches[k].set_nn(X_patches[indices[k, j]])
                                break

                    elif len(Z_patches[k].nn_patch) == K and R > 0:
                        # the nn query is already sorted by distances
                        for j in range(NNF.shape[0]):
                            # Loop until we find an unused X patch
                            if X_patches[indices[k, j]].nn_patch is None:
                                Z_patches[k].set_nn(X_patches[indices[k, j]])
                                R -= 1
                                break

            out[:, :, 0:4] = updated_image
            # If updated neighbors is the same as current neighbors, then exit the loop and set the output image to the
            # updated image

        # update "prev frame" rgb and alpha for next frame's use
        t = SimilarityTransform(translation=(0, 20)) # vector field: moves up 20
        X_prev = warp(out[:, :, 0:4], t) # warps the output according to transformation
        # save the image
        fname = 'output{0}.jpg'.format(frame_num)
        im = out[:, :, 0:4].astype(np.uint8)
        skio.imsave(fname, im)

        # display the image
        skio.imshow(im)
        skio.show()


main()
