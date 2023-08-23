import cv2
import numpy as np


def nnUpsample(img):
  h, w, d = img.shape
  newh, neww = h*4, w*4
  upsampleImg = np.zeros((newh, neww, d))
  for dd in range(d):
    for y in range(newh):
      for x in range(neww):
        oriy = int(np.floor(y/4.0))
        orix = int(np.floor(x/4.0))
        upsampleImg[y, x, dd] = img[oriy, orix, dd]
  return upsampleImg


def bilinearUpsample(img):
  h, w, d = img.shape
  newh, neww = h*4, w*4
  upsampleImg = np.zeros((newh, neww, d))
  for dd in range(d):
    for y in range(newh):
      for x in range(neww):
        oriy = y/4.0
        orix = x/4.0
        upsampleImg[y, x, dd] = bilinearInterpolation(img, orix, oriy, dd)
  return upsampleImg

def bilinearInterpolation(img, x, y, d):
  h, w, dep = img.shape
  x1, y1, x2, y2 = int(np.floor(x)), int(np.floor(y)), int(np.floor(x))+1, int(np.floor(y))+1

  if x1<0 or x2>=img.shape[1] or y1<0 or y2>=img.shape[0]:
    return img[max(0, min(h-1, int(np.round(y)))), max(0, min(w-1, int(np.round(x)))), d]
  
  q11, q12, q21, q22 = img[y1, x1, d], img[y2, x1, d], img[y1, x2, d], img[y2, x2, d]
  w1 = (x2 - x) * (y2 - y)
  w2 = (x - x1) * (y2 - y)
  w3 = (x2 - x) * (y - y1)
  w4 = (x - x1) * (y - y1)
  interpolationValue = (w1*q11 + w2*q21 + w3*q12 + w4*q22)
  return interpolationValue

if __name__ == "__main__":
  img = cv2.imread('./img/bee.jpeg')
  nnImg = nnUpsample(img)
  linearImg =  bilinearUpsample(img)
  cv2.imwrite('./out/bee_linear.jpg', linearImg)
  cv2.imwrite('./out/bee_near.jpg', nnImg)
