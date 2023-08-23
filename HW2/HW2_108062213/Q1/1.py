import numpy as np
import cv2


def dct2(block):
    n = 8
    dct = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            cv = 1 / np.sqrt(2) if v == 0 else 1
            for i in range(n):
                for j in range(n):
                    dct[u,v] += 0.25 * cu * cv * block[i,j] * np.cos((2*i+1)*u*np.pi/16) * np.cos((2*j*v*np.pi/16))
    return dct

def idct2(dct):
    n = 8
    block = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for u in range(n):
                for v in range(n):
                    cu = 1 / np.sqrt(2) if u == 0 else 1
                    cv = 1 / np.sqrt(2) if v == 0 else 1
                    block[i,j] += 0.25 * cu * cv * dct[u,v] * np.cos((2*i+1)*u*np.pi/16) * np.cos((2*j+1)*v*np.pi/16)
    return block


def compressed(image, n, m, table):
    if len(image.shape) == 2:
        h, w = image.shape
        d = 1
        image = np.expand_dims(image, axis=-1)
    else:
        h, w, d = image.shape
    # divide the image into 8x8 pixels
    num_blocks_h = h // 8
    num_blocks_w = w // 8
    max_value = np.array([-1,-1,-1])
    min_value = np.array([260,260,260])


    dct_blocks = np.zeros((h, w, d))
    quantized_blocks = np.zeros((h, w, d))
    reconstructed_image = np.zeros((h, w, d))
    for channel in range(d):
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                #subtract 128 from each pixel to fit dct domain
                block = image[i*8:(i+1)*8, j*8:(j+1)*8, channel]
                dct = dct2(block)
                # lower-frequency coefficients
                dct[n:, :] = 0
                dct[:, n:] = 0
                dct_blocks[i*8:(i+1)*8, j*8:(j+1)*8, channel] = dct
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                dct = dct_blocks[i*8:(i+1)*8, j*8:(j+1)*8, channel]
                q_block = np.round(dct / table)
                if(q_block[:n,:n].max() > max_value[channel]):
                    max_value[channel] = q_block[:n,:n].max()
                if(q_block[:n,:n].min() < min_value[channel]):
                    min_value[channel] = q_block[:n,:n].min()
                quantized_blocks[i*8:(i+1)*8, j*8:(j+1)*8, channel] = q_block
        step = (max_value[channel] - min_value[channel])/(2**m-1)
        quantized_blocks = np.round(quantized_blocks / step) * step
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = quantized_blocks[i*8:(i+1)*8, j*8:(j+1)*8, channel]
                block *= table
                block = idct2(block)
                block = block.astype(np.int16)
                reconstructed_image[i*8:(i+1)*8, j*8:(j+1)*8, channel] = block

    if d == 1:
        reconstructed_image = np.squeeze(reconstructed_image, axis=-1)

    return reconstructed_image

def rgb2ycbcr(image):
    ycbcr = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            ycbcr[i, j, 0] = 16 + 0.257*r + 0.564*g + 0.098*b
            ycbcr[i, j, 1] = 128 - 0.148*r - 0.291*g + 0.439*b
            ycbcr[i, j, 2] = 128 + 0.439*r - 0.368*g - 0.071*b
    return ycbcr

def ycbcr2rgb(ycbcrArr):
    rgb = np.zeros_like(ycbcrArr, dtype=np.float32)
    for i in range(ycbcrArr.shape[0]):
        for j in range(ycbcrArr.shape[1]):
            y, cb, cr = ycbcrArr[i, j]
            rgb[i, j, 0] = 1.164*(y-16) + 1.596*(cr-128)
            rgb[i, j, 1] = 1.164*(y-16) - 0.382*(cb-128) - 0.813*(cr-128)
            rgb[i, j, 2] = 1.164*(y-16) + 2.017*(cb-128)
    return rgb

def upsampling(y, cb, cr):
    upsampled_cb = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1)
    upsampled_cr = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1)
    upsampled_ycbcr = np.stack((y, upsampled_cb, upsampled_cr), axis=-1)
    return upsampled_ycbcr

def subsampling(ycbcr):
    y = ycbcr[:,:,0]
    cb, cr = ycbcr[::2,::2,1], ycbcr[::2,::2,2]
    return y, cb, cr

def psnr(original, compressed):
    mse = np.mean((original - compressed)**2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

if __name__ == '__main__':
    quantized_table = np.array([[8,6,6,7,6,5,8,7],
                                [7,7,9,9,8,10,12,20],
                                [13,12,11,11,12,25,18,19],
                                [15,20,29,26,31,30,29,26],
                                [28,28,32,36,46,39,32,34],
                                [44,35,28,28,40,55,41,44],
                                [48,49,52,52,52,31,39,57],
                                [61,56,50,60,46,51,52,50]])

    luminance_table = np.array([[16, 12, 14, 14, 18, 24, 49, 72],
                                [11, 12, 13, 17, 22, 35, 64, 92],
                                [10, 14, 16, 22, 37, 55, 78, 95],
                                [16, 19, 24, 29, 56, 64, 87, 98],
                                [24, 26, 40, 51, 68, 81, 103, 112],
                                [40, 58, 57, 87, 109, 104, 121, 100],
                                [51, 60, 69, 80, 103, 113, 120, 103],
                                [61, 55, 56, 62, 77, 92, 101, 99]])

    chrominance_table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                  [18, 21, 26, 66, 99, 99, 99, 99],
                                  [24, 26, 56, 99, 99, 99, 99, 99],
                                  [47, 66, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99]])
    # part a
    # for filename in (['cat.jpeg', 'Barbara.jpeg']):
    #     image = cv2.imread(filename)
    #     for n in [2,4]:
    #         for m in [4,8]:
    #             reconstructed_image = compressed(image, n, m, quantized_table)
    #             reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    #             print(f'PSNR_{filename}_n{n}_m{m}_a: ',psnr(image, reconstructed_image))
    #             cv2.imwrite(f'./output/{filename[:-5]}_n{n}m{m}_a.jpg', reconstructed_image)
    # part b
    for filename in (['cat.jpeg', 'Barbara.jpeg']):
        image = cv2.imread(filename)
        y, cb, cr = subsampling(rgb2ycbcr(image))
        for n in [2,4]:
            for m in [4,8]:
                com_y = compressed(y, n, m, luminance_table)
                com_cb = compressed(cb, n, m, chrominance_table)
                com_cr = compressed(cr, n, m, chrominance_table)
                com_y = np.clip(com_y, 0, 255).astype(np.uint8)
                com_cb = np.clip(com_cb, 0, 255).astype(np.uint8)
                com_cr = np.clip(com_cr, 0, 255).astype(np.uint8)
                upsampled_ycbcr = upsampling(com_y, com_cb, com_cr)
                reconstructed_rgb = ycbcr2rgb(upsampled_ycbcr)
                reconstructed_rgb = np.clip(reconstructed_rgb, 0, 255).astype(np.uint8)
                print(f'PSNR_{filename}_n{n}_m{m}_b: ',psnr(image, reconstructed_rgb))
                cv2.imwrite(f'./output/{filename[:-5]}_n{n}m{m}_b.jpg', reconstructed_rgb)