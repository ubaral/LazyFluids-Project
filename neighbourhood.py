import numpy as np
import skimage.io as skio

def nbhd_dist(n, z_rgb, x_rgb, z_a, x_a, z_blur, x_blur, z_rgb_prev=None, x_rgb_prev=None, z_a_prev=None, x_a_prev=None):
  """Takes in two neighbourhoods' matrices and returns a value representing the distance

  Args:
    n: blur alpha mask scaling
    z_rgb: nbhd 1's rgb matrix - nbhd width x nbhd height x 3
    x_rgb: nbhd 2's rgb matrix - nbhd width x nbhd height x 3
    z_a: nbhd 1's alpha mask - nbhd width x nbhd height
    x_a: nbhd 2's alpha mask - nbhd width x nbhd height
    z_blur: nbhd 1's blurred alpha mask - nbhd width x nbhd height
    x_blur: nbhd 2's blurred alpha mask - nbhd width x nbhd height
    z_rgb_prev (optional): rgb of previous nbhd 1 frame: for videos
    x_rgb_prev (optional): rgb of previous nbhd 2 frame: for videos
    z_a_prev (optional): alpha mask of previous nbhd 1 frame: for videos
    x_a_prev (optional): alpha mask of previous nbhd 2 frame: for videos

  Returns:
    dist: distance (measurement of difference) between nbhd1 and nbhd2
  """
  # assuming equal width and height in patches
  w = len(z_rgb)
  d = 0
  # r, g, and b:
  d += np.linalg.norm(np.linalg.norm(z_rgb-x_rgb, axis=0))**2
  # alpha:
  d += 3*(np.linalg.norm(z_a - x_a)**2)
  # blur:
  d += n*(np.linalg.norm(z_blur - x_blur)**2)
  # previous, if they exist
  if z_rgb_prev != None:
    d += np.linalg.norm(np.linalg.norm(z_rgb_prev-x_rgb_prev, axis=0))**2
    d += 3*(np.linalg.norm(z_a_prev - x_a_prev)**2)
  return d