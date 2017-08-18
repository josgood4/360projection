import cv2
import numpy as np
import scipy.misc

import equirectangular as e

def xrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def yrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def equi_to_barrel(z_offset, theta, phi, width, img):
  """
  theta is pitch
  phi is yaw
  width is the number of horizontal pixels in the destination image
  height is determined assuming a 5:2 aspect ratio
  """
  # currently does not take theta and phi into account
  (base_height, base_width, _) = img.shape
  
  height = int(width * 2 / 5)
  r = width//10

  new_img = np.zeros((height, width, 3), np.uint8)

  # generate a 'circle mask,' bools
  dx, dy = np.meshgrid(np.arange(-r, r), np.arange(-r, r))
  bools = dx**2 + dy**2 <= r**2
  bools = np.expand_dims(bools, axis=2)

  zeros = np.zeros((r * 2, r * 2, 3), np.int)

  # render and copy an equirectangular view,
  #   but only inside the circle described by bools
  barrel_top = e.render_image_np(np.pi/2, 0, np.pi/2, np.pi/2, 2 * r, img)
  new_img[:2 * r, width*4//5:] = np.where(bools, barrel_top, zeros)
  
  barrel_bottom = e.render_image_np(-np.pi/2, 0, np.pi/2, np.pi/2, 2 * r, img)
  new_img[2 * r:, width*4//5:] = np.where(bools, barrel_bottom, zeros)

  # equirectangular part:
  new_img[:,:width*4//5] = scipy.misc.imresize(img, (height*2, width*4//5))[height//2:height*3//2]

  return new_img
