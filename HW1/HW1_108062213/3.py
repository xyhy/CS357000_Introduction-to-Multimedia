import numpy as np
import matplotlib.pyplot as plt

import util

def prblm_a(rgb):
  yiqArr = util.RGB2YIQ(rgb)
  y = yiqArr[:,:,0]
  hist = util.Hist(y)
  
  fig, ax= plt.subplots()
  ax.bar(range(256), hist)
  ax.set_title('Histogram of Y channel')
  plt.savefig('./out/Y_hist.jpg')
  plt.clf()
  
  return yiqArr

def prblm_b(YIQ):
  y = YIQ[:,:,0]
  yGamma = util.GammaTransform(y, 4.0)
  gammaHist = util.Hist(yGamma)
  
  fig, ax= plt.subplots()
  ax.bar(range(256), gammaHist)
  ax.set_title('Histogram of transformed Y channel')
  plt.savefig('./out/Y_gamma_hist.jpg')
  plt.clf()
  
  return yGamma

def prblm_c(yiq):
  rgbArr = util.YIQ2RGB(yiq)
  height, width, _ = rgbArr.shape
  fig, ax= plt.subplots(figsize=(width/100, height/100))
  ax.imshow(rgbArr/255.0)
  ax.set_title('Transformed Image')
  ax.axis('off')
  plt.savefig('./out/gamma_img.jpg')


if __name__ == "__main__":
  img = plt.imread('./img/lake.jpeg')
  yiq = prblm_a(img)
  yGamma = prblm_b(yiq)
  newYiq = yiq
  newYiq[:,:,0] = yGamma
  prblm_c(newYiq)