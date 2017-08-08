import cv2
import numpy as np
import sys
#import left_eye


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


def render_image(theta0, phi0, fov_h, fov_v, width, img):
  """
  theta0 is pitch
  phi0 is yaw
  """
  m = np.dot(yrotation(phi0),xrotation(theta0))
  
  (base_height, base_width, _) = img.shape

  # for a barrel layout, only the first 4/5 of it are like equi

  scaled_base_width = 0.8*base_width

  height = int(width * np.tan(fov_v/2.0) / np.tan(fov_h/2.0))

  new_img = np.zeros((height, width, 3), np.uint8)

  for diy in np.arange(height):
    y = np.tan(fov_v/2.) - diy*np.tan(fov_v/2.)/float(height) * 2.

    for dix in np.arange(width):
      x = dix*np.tan(fov_h/2.)/float(width) * 2. - np.tan(fov_h/2.)

      v = np.dot(m,np.array([x, y, 1.]))
      
      diag = np.sqrt(v[2]**2 + v[0]**2)
      theta = np.pi/2 - np.arctan2(v[1],diag)
      phi = np.arctan2(v[0],v[2]) + np.pi

      ##sys.stdout.write("%f\t %f\n" % (theta, phi))
      ##sys.stdout.flush()
      
      if theta < (np.pi/4):
        r = 0 if theta==0 else abs(base_height * 0.25 * np.tan(theta))
        by = int(-r * np.sin(phi+np.pi/2) + base_height * 0.25) 
        bx = int(r * np.cos(phi+np.pi/2) + base_width * 0.9)
        if bx>=base_width: continue
        new_img[diy, dix] = img[by, bx]

      elif theta < np.pi-(np.pi/4):
        by = int((theta-(np.pi/4))*base_height/(np.pi-2*(np.pi/4)))
        bx = int(phi*scaled_base_width/(2*np.pi))
        if bx>=scaled_base_width: continue
        new_img[diy, dix] = img[by, bx]
      
      else:
        r = 0 if theta==np.pi else abs(base_height * 0.25 * np.tan(np.pi-theta))
        by = int(r * np.sin(phi+np.pi/2) + base_height * 0.75) 
        bx = int(r * np.cos(phi+np.pi/2) + base_width * 0.9)
        if bx>=base_width or by>=base_height: continue
        new_img[diy, dix] = img[by, bx]

      ##new_img[diy, dix] = img[by, bx]      

          
  return new_img


def deg2rad(d):
  return float(d) * np.pi / 180


if __name__== '__main__':
 
  STEREO = False
  img_file = 'b_2500.png'
  img = cv2.imread(img_file)
  
  #if STEREO:
    #img = left_eye.extract_left(image)

  face_size = 1000
  
  yaw = 0
  pitch = 0
  
  fov_h = 90
  fov_v = 90
  
  rimg = render_image(deg2rad(pitch), deg2rad(yaw), \
                      deg2rad(fov_v), deg2rad(fov_h), \
                      face_size, img)
  cv2.imwrite('%s_%d_%d.bmp'%(img_file[:img_file.find('.')], yaw, pitch), rimg) #switched yaw and pitch so it would match other code

#   equi_to_cube(600,img)
