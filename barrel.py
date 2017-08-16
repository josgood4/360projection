import cv2
import numpy as np
import sys


def rotate_image(old_image):
  (old_height, old_width, _) = old_image.shape
  M = cv2.getRotationMatrix2D(((old_width-1)/2.,(old_height-1)/2.),270,1)
  rotated = cv2.warpAffine(old_image,M,(old_width, old_height))
  return rotated


def xrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def yrotation(th):
  c = np.cos(th)
  s = np.sin(th)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def render_image_np(theta0, phi0, fov_h, fov_v, width, img):
  """
  theta0 is pitch
  phi0 is yaw
  render view at (pitch, yaw) with fov_h by fov_v
  width is the number of horizontal pixels in the view
  """
  m = np.dot(yrotation(phi0), xrotation(theta0))
  
  (base_height, base_width, _) = img.shape

  height = int(width * np.tan(fov_v / 2) / np.tan(fov_h / 2))

  new_img = np.zeros((height, width, 3), np.uint8)
    
  DI = np.ones((height * width, 3), np.int)
  trans = np.array([[2.*np.tan(fov_h / 2) / float(width), 0., -np.tan(fov_h / 2)],
                    [0., -2.*np.tan(fov_v / 2) / float(height), np.tan(fov_v / 2)]])
  
  xx, yy = np.meshgrid(np.arange(width), np.arange(height))
  
  DI[:, 0] = xx.reshape(height * width)
  DI[:, 1] = yy.reshape(height * width)

  v = np.ones((height * width, 3), np.float)

  v[:, :2] = np.dot(DI, trans.T)
  v = np.dot(v, m.T)
  
  diag = np.sqrt(v[:, 2] ** 2 + v[:, 0] ** 2)
  theta = np.pi / 2 - np.arctan2(v[:, 1], diag)
  theta_e = np.pi / 2 - np.arctan2(v[:, 1], diag) * 2
  phi = np.arctan2(v[:, 0], v[:, 2]) + np.pi

  ey = np.rint(theta_e * base_height / np.pi).astype(np.int)
  ex = np.rint(phi * base_width * 0.8 / (2 * np.pi)).astype(np.int)

  ex[ex >= base_width * 0.8] = base_width * 0.8 - 1
  ey[ey >= base_height] = base_height - 1

  # top of barrel
  if theta0 > 0:
    condition = theta < np.pi/4
    origin = (base_width * 0.9, base_height * 0.25)
    neg = 1
  # bottom of barrel
  else:
    condition = theta > np.pi*3/4
    origin = (base_width * 0.9, base_height * 0.75)
    neg = -1

  r = base_height * 0.25 * np.tan(theta[condition]) \

  ex[condition] = r * neg * np.cos(phi[condition]+np.pi/2) + origin[0]
  ey[condition] = -r * np.sin(phi[condition]+np.pi/2) + origin[1]

  ex[ex >= base_width] = base_width - 1
  
  new_img[DI[:, 1], DI[:, 0]] = img[ey, ex]
  return new_img



def deg2rad(d):
  return float(d) * np.pi / 180


if __name__== '__main__':
 
  img_file = 'b_2500.png'
  img = cv2.imread(img_file)

  width = 1000
  
  yaw = 0
  pitch = 0
  
  fov_h = 90
  fov_v = 90
  
  rimg = render_image(deg2rad(pitch), deg2rad(yaw), \
                      deg2rad(fov_v), deg2rad(fov_h), \
                      width, img)
  cv2.imwrite('%s_%d_%d.bmp'%(img_file[:img_file.find('.')], yaw, pitch), rimg) #switched yaw and pitch so it would match other code

