import numpy as np
from PIL import Image
import cv2

import util

def prblm_a(img, bits):
  pixelArray = np.copy(img)
  pixelArray = pixelArray.reshape((-1,3))
  cubes = [pixelArray]
  while len(cubes)<2**bits:
    newCubes = []
    for cube in cubes:
      leftCube, rightCube = util.SplitCube(cube)
      newCubes.append(leftCube)
      newCubes.append(rightCube)
    cubes = newCubes
  colors = [util.MedianColor(cube) for cube in cubes]
  # newPixelsArray = np.array([util.ClosestColor(pixel, colors) for pixel in pixelArray])
  newRgbArray = np.zeros_like(img)
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      pixel = img[y, x]
      newRgbArray[y, x] = util.ClosestColor(pixel, colors)
      
  # newRgbArray = newPixelsArray.reshape(img.shape)
  mse = util.MSE(img, newRgbArray)
  return newRgbArray, mse, colors

def prblm_b(img, colors):
  h, w, d = img.shape
  oriImg = np.copy(img).astype(float)
  ditherImg = np.zeros_like(img)
  for y in range(h):
    for x in range(w):
      pixel = oriImg[y, x]
      newPixel = util.ClosestColor(pixel, colors)
      ditherImg[y, x] = newPixel
      err = pixel - newPixel

      if x < w - 1:
        oriImg[y, x + 1] = oriImg[y, x+1] + err * 7 / 16
      if x > 0 and y < h - 1:
        oriImg[y + 1, x - 1] = oriImg[y+1, x-1] + err * 3 / 16
      if y < h - 1:
        oriImg[y + 1, x] = oriImg[y+1, x] + err * 5 / 16
      if x < w - 1 and y < h - 1:
        oriImg[y + 1, x + 1] = oriImg[y+1, x+1] + err * 1 / 16
  mse = util.MSE(img.astype(np.uint8), ditherImg.astype(np.uint8))
  return ditherImg.astype(np.uint8), mse


if __name__ == "__main__":
  originalImg = Image.open('./img/Lenna.jpeg')
  rgbArray = np.array(originalImg)
  # originalImg = cv2.imread('./img/Lenna.jpeg')
  # rgbArray = np.array(originalImg)
  # testImg = Image.open('./img/lake.jpeg')
  # testrgbArray = np.array(testImg)
  # testbit3MedianCut, testmmse3, testcolors3=prblm_a(testrgbArray, 3)
  # outputImg = Image.fromarray(testbit3MedianCut)
  # outputImg.save("./exp/median_cut3.png")
  # testbit3Dither, testdmse3 = prblm_b(testrgbArray, testcolors3)
  # outputImg = Image.fromarray(testbit3Dither)
  # outputImg.save("./exp/error_diffusion_dithering_3.png")
  # print("MSE for 3 bits: ", testmmse3)
  # print("MSE for 3 bits: ", testdmse3)

  bit3MedianCut, mmse3, colors3=prblm_a(rgbArray, 3)
  bit6MedianCut, mmse6, colors6=prblm_a(rgbArray, 6)
  outputImg = Image.fromarray(bit3MedianCut)
  outputImg.save("./out/median_cut3.png")
  outputImg = Image.fromarray(bit6MedianCut)
  outputImg.save("./out/median_cut6.png")
  print("MSE for 3 bits: ", mmse3)
  print("MSE for 6 bits: ", mmse6)
  
  bit3Dither, dmse3 = prblm_b(rgbArray, colors3)
  bit6Dither, dmse6 = prblm_b(rgbArray, colors6)
  outputImg = Image.fromarray(bit3Dither)
  outputImg.save("./out/error_diffusion_dithering_3.png")
  outputImg = Image.fromarray(bit6Dither)
  outputImg.save("./out/error_diffusion_dithering_6.png")
  print("MSE for 3 bits: ", dmse3)
  print("MSE for 6 bits: ", dmse6)

