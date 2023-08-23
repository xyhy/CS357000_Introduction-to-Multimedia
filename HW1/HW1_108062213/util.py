import numpy as np
import math
from scipy.spatial import distance

def RGB2YIQ(img):
  transform = np.array([[0.299, 0.587, 0.114],
                          [0.596, -0.274, -0.322],
                          [0.211, -0.523, 0.312]])
  shape = img.shape
  yiq = np.dot(img.reshape(-1,3), transform.T).reshape(shape)
  return yiq


def YIQ2RGB(img):
  transform = np.linalg.inv(np.array([[0.299, 0.587, 0.114],
                                      [0.596, -0.274, -0.322],
                                      [0.211, -0.523, 0.312]]).transpose())
  shape = img.shape
  rgb = np.dot(img.reshape(-1,3), transform).reshape(shape)
  return rgb


def Hist(channel):
  histogram = np.zeros(256)
  for i in range(channel.shape[0]):
    for j in range(channel.shape[1]):
      intensity = math.floor(channel[i][j])
      histogram[intensity] += 1
  return histogram


def GammaTransform(img, gamma):
  imgNorm = img.astype(float) / 255
  imgGamma = np.power(imgNorm, gamma)
  imgGamma *= 255.0
  return imgGamma.astype(np.uint8)





def SplitCube(cube):
  ranges = np.max(cube, axis=0) - np.min(cube, axis=0)
  dim = np.argmax(ranges)
  sortedcube = cube[cube[:, dim].argsort()]
  medianidx = len(cube) // 2
  leftcube = sortedcube[:medianidx]
  rightcube = sortedcube[medianidx:]
  return leftcube, rightcube


def MedianColor(cube):
  return np.array(np.mean(cube, axis=0).astype(int))
  
def ClosestColor(color, color_list):
  idx = np.argmin(distance.cdist([color], color_list, metric='euclidean'))
  return color_list[idx]

def MSE(oriImg, newImg):
  return np.mean((oriImg - newImg) ** 2)