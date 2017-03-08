'''
Title: Lazy Fluids Project
File name: main.py
Author: Utsav Baral
Date created: 3/6/2017
Python Version: 3.5
'''

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from scipy import ndimage
import random


# Function that takes in an image, a pixel within the image, and a nbhd width. Returns an array containing the nbhd of
# pixels
def get_nbhd(img, pix_coord, width):
    pass

# Function that takes in two patches of pixels and returns the error between the two pixels
def calculate_error(patch1, patch2):
    pass

# name of the input file
imname = 'texture.jpg'

# read in the image
Z_src = skio.imread(imname)

nbhd_width = 32

# output image X, size is 800 by 800
out = np.ndarray((800,800,3))

# Create the set of points that are the centers of the neighborhoods we will be comparing
x_centers = np.arange(nbhd_width/2, out.shape[0] - nbhd_width/2, nbhd_width/4)
y_centers = np.arange(nbhd_width/2, out.shape[1] - nbhd_width/2, nbhd_width/4)

# Cartesian product of the x and y partitions
prod = np.transpose([np.tile(x_centers, len(y_centers)), np.repeat(y_centers, len(x_centers))])
nbhd_centers = set([tuple(prod[i]) for i in range(len(prod))])

output_patch_nearest_neigbors = dict.fromkeys(nbhd_centers)

# Initialize the nearest neighbors for each nbhd in X(output) to random nbhds in Z(source)
for patch in output_patch_nearest_neigbors:
    # x_rand = random.randrange(0, Z_src.shape[0])
    # y_rand = random.randrange(0, Z_src.shape[1])
    x_rand = random.randrange(nbhd_width/2, Z_src.shape[0] - nbhd_width/2 + 1)
    y_rand = random.randrange(nbhd_width/2, Z_src.shape[1] - nbhd_width/2 + 1)
    output_patch_nearest_neigbors[patch] = (x_rand, y_rand)

N = 10 # number of iterations
for i in range(N):
    # update the output image to be the pixel values that minimize the Energy Function. Should calculate as a system of
    # linear equations, Ax = b. Removed previous energy for
    pass

# save the image
#fname = 'output.jpg'
#skio.imsave(fname, im)

# display the image
skio.imshow(Z_src)
skio.show()
