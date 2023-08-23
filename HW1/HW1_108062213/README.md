# 108062213 顏浩昀 HW1



### Problem 1
### Discription
>This code implements two functions for image processing. First one takes an RGB image and the number of bits to reduce the color palette to using median cut algorithm, and it calculate the mean squared error (MSE) between the original and the new image. Second one applies error diffusion dithering algorithm to the image and calculate the MSE between the original and the new image.The output images are saved in the './out/' directory.
### Requirements
> 1. python3 == 3.7.15
> 2. numpy == 1.21.6
> 3. opencv-python == 4.6.0.66
### Usage
>python3 1.py
### Notes
>The input image should be in JPEG format.
The output images will be saved in PNG format.
---
## Problem 2
### Discription
>This Python script performs image upsampling using nearest-neighbor and bilinear interpolation techniques. The output images are saved in the './out/' directory.
### Requirements
>This script requires the following libraries:
>1. python3 == 3.7.15
>2. numpy == 1.21.6
>3. opencv-python == 4.6.0.66
### Usage
>python3 2.py
### Notes
>The input image should be in JPEG format.
The output images will be saved in PNG format.
---
## Problem 3
### Discription
>This code is designed to perform color space transformations and gamma correction. Specifically, it converts the input image from RGB to YIQ color space, computes the histogram of the Y channel of the image, performs gamma correction on the Y channel, and then converts the image back to RGB color space. The output images are saved in the './out/' directory.
### Requirements
>1. python3 == 3.7.15
>2. numpy == 1.21.6
>3. matplotlib == 3.5.3
### Usage
>python3 3.py
### Notes
>The input image should be in JPEG format.
The output images will be saved in PNG format.