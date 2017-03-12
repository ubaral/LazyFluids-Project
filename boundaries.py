import numpy as np
import skimage as sk
import skimage.filters as skf
import skimage.io as skio

class boundaries(object):
  """extracts useful boundary features from images
  note: we add things like methods to retrieve neighbourhoods here as well if necessary
  """

  def __init__(self, blur_alpha, boundary_image, boundary_list, inner_image, inner_list):
    self.__blur_alpha = blur_alpha
    self.__boundary_img = boundary_image
    self.__boundary_list = boundary_list
    self.__inner_img = inner_image
    self.__inner_list = inner_list

  def get_blur_alpha(self):
    """returns gaussian blurred alpha mask"""
    return self.__blur_alpha
  def get_boundary_img(self):
    """returns image with the boundaries as black on white background"""
    return self.__boundary_img
  def get_boundary_list(self):
    """returns list of all boundary pixels"""
    return self.__boundary_list
  def get_inner_img(self):
    """returns list of all inner texture pixels"""
    return self.__inner_img
  def get_inner_list(self):
    return self.__inner_list
  blur_alpha = property(get_blur_alpha)
  boundary_img = property(get_boundary_img)
  boundary_list = property(get_boundary_list)
  inner_img = property(get_inner_img)
  inner_list = property(get_inner_list)


def separate_boundaries(alpha_file_name, blur_r, opacity_l, opacity_u):
  """Separates boundary of image

  Args:
    alpha_file_name: name of image file (string)
    blur_r: sigma for gaussian blur (float)
    opacity_l: lower opacity limit for boundary (float, 0-1)
    opacity_u: upper opacity limit for boundary (float, 0-1)
  Returns:
    boundaries: boundary object, allowing retrieval of the boundary blur mask and boundary/inner in image or list form
  """
  # read input file
  alpha = skio.imread(alpha_file_name)
  # gaussian blur
  blur_alpha = skf.gaussian(image=alpha, sigma=blur_r, multichannel=False)
  # getting separate boundary and inner images and lists
  width = len(alpha)
  height = len(alpha[0])
  boundary_image = np.zeros(shape=(width, height, 3))
  boundary_list = []
  inner_image = np.zeros(shape=(width, height, 3))
  inner_list = []
  for x in range(width):
    for y in range(height):
      p = blur_alpha[x][y][0]
      # getting boundary pixels
      if p >= opacity_u or p <= opacity_l:
        boundary_image[x][y] = np.array([1, 1, 1])
        boundary_list.append((x, y))
      # getting inner pixels
      if p >= opacity_l:
        inner_image[x][y] = np.array([1, 1, 1])
        inner_list.append((x, y))
  return boundaries(blur_alpha, boundary_image, boundary_list, inner_image, inner_list)


# example on how to use separate_boundaries
thing = separate_boundaries("texture2.jpg", 6, .01, 0.95) # separates with blur sigma=6, with all pixels between .01 and .95 as boundary pixels
print(thing.inner_list) # example for getting list of all pixels of texture not in boundary
skio.imshow(thing.boundary_img) # example for getting boundary images
skio.show()