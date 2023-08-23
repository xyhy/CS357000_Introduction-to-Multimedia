import numpy as np
import cv2
import math
import time


# Define the full search method
def full_search(reference, target, block_size, search_range):
    h, w, c = reference.shape
    predicted_img = np.zeros_like(target)
    motion_vectors = np.zeros((h//block_size, w//block_size, 2), dtype=int)
    residuals_img = np.zeros_like(target)

    for row in range(0, h, block_size):
        for col in range(0, w, block_size):
            center = (row, col)
            block = reference[row:row+block_size, col:col+block_size, :]
            mv, sad = full_search_helper(block, center, target, search_range)
            motion_vectors[row//block_size, col//block_size] = mv
            predicted_img[row:row+block_size,col:col+block_size, :] = target[row+mv[0]:row+mv[0]+block_size,col+mv[1]:col+mv[1]+block_size, :]
    residuals_img = cv2.absdiff(predicted_img, target)
    
    return predicted_img, motion_vectors, residuals_img

def full_search_helper(block, center, target, search_range):
    x, y = center
    best_mv = [0, 0]
    min_sad = SAD(block, target[x:x+block.shape[0], y:y+block.shape[1], :])
    for dx in range(-search_range, search_range+1):
        for dy in range(-search_range, search_range+1):
            i, j = x+dx, y+dy
            if i<0 or i+block.shape[0]>=target.shape[0] or j<0 or j+block.shape[1]>=target.shape[1]:
                continue
            target_block = target[i:i+block.shape[0], j:j+block.shape[1], :]
            sad = SAD(block, target_block)
            if sad < min_sad:
                min_sad = sad
                best_mv = [dx, dy]
    return best_mv, min_sad


# Define the 2D logarithmic search method
def log_search(reference, target, block_size, search_range):
    h, w, c = reference.shape
    predicted_img = np.zeros_like(target)
    motion_vectors = np.zeros((h//block_size, w//block_size, 2), dtype=int)
    residuals_img = np.zeros_like(target)
    
    for row in range(0, h, block_size):
        for col in range(0, w, block_size):
            block = reference[row:row+block_size, col:col+block_size, :]
            center = (row, col)
            mv, sad = log_search_helper(block, center, target, search_range)
            motion_vectors[row//block_size, col//block_size] = mv
            predicted_img[row:row+block_size,col:col+block_size, :] = target[row+mv[0]:row+mv[0]+block_size,col+mv[1]:col+mv[1]+block_size, :]
    residuals_img = cv2.absdiff(predicted_img, target)
    return predicted_img, motion_vectors, residuals_img

def log_search_helper(block, center, target, search_range):
    x, y = center
    dx, dy = 0, 0
    steps = int(math.ceil(math.log2(search_range)))
    for step in range(steps, 0, -1):
        best_mv = [0, 0]
        min_sad = SAD(block, target[x:x+block.shape[0], y:y+block.shape[1], :])
        for dxs in range(-1, 2):
            for dys in range(-1, 2):
                i = x+dx+(dxs<<(step-1))
                j = y+dy+(dys<<(step-1))
                if i<0 or i+block.shape[0]>=target.shape[0] or j<0 or j+block.shape[1]>=target.shape[1]:
                    continue
                target_block = target[i:i+block.shape[0], j:j+block.shape[1], :]
                sad = SAD(block, target_block)
                if sad < min_sad:
                    min_sad = sad
                    best_mv = [i-x, j-y]
        dx, dy = best_mv[0], best_mv[1]
    return best_mv, min_sad


def SAD(ref, target):
    return np.sum(np.abs(ref - target))

def PSNR(origin, predict):
    mse = np.mean((origin - predict) ** 2)
    if mse == 0:
        return 100
    else:
        return 20*np.log10(255/np.sqrt(mse))

def drawArrowLine(vector_image, vectors, block_size):
    for row in range(0, vector_image.shape[0], block_size):
        for col in range(0, vector_image.shape[1], block_size):
                x, y = vectors[row//block_size, col//block_size]
                pt1 = (col+block_size//2, row+block_size//2)
                pt2 = (col+y+block_size//2, row+x+block_size//2)
                cv2.arrowedLine(vector_image, pt1, pt2, (0, 0, 255), 1)
    return vector_image


if __name__ == '__main__':
    # Load the images
    ref_image = cv2.imread('./img/40.jpg')
    target_image = cv2.imread('./img/42.jpg')
    test_image = cv2.imread('./img/51.jpg')
    
    # parameters
    macroblock_sizes = [8, 16]
    search_ranges = [8, 16]
    
    # Q1
    for macroblock_size in macroblock_sizes:
        for search_range in search_ranges:
            # full search
            motion_vector_full_image = target_image.copy()
            start = time.time()
            predicted_image_full, motion_vector_full, residual_image_full = full_search(ref_image, target_image, macroblock_size, search_range)
            end = time.time()
            motion_vector_full_image = drawArrowLine(motion_vector_full_image, motion_vector_full, macroblock_size)
            sad_full = SAD(target_image, predicted_image_full)
            psnr_full = PSNR(target_image, predicted_image_full)
            cv2.imwrite(f'./out/full_predicted_r{search_range}_b{macroblock_size}.jpg', predicted_image_full)
            cv2.imwrite(f'./out/full_motion_vector_r{search_range}_b{macroblock_size}.jpg', motion_vector_full_image)
            cv2.imwrite(f'./out/full_residual_r{search_range}_b{macroblock_size}.jpg', residual_image_full)
            print('full_sad_value_r_{}_b_{}: {}'.format(search_range, macroblock_size, sad_full))
            print('full_psnr_r_{}_b_{}: {}'.format(search_range, macroblock_size, psnr_full))
            print('full_time_r_{}_b_{}: {} seconds'.format(search_range, macroblock_size, end-start))
            
            
            # 2d log search
            motion_vector_log_image = target_image.copy()
            start = time.time()
            predicted_image_2d, motion_vector_2d, residual_image_2d = log_search(ref_image, target_image, macroblock_size, search_range)
            end = time.time()
            motion_vector_log_image = drawArrowLine(motion_vector_log_image, motion_vector_2d, macroblock_size)
            sad_2d = SAD(ref_image, predicted_image_2d)
            psnr_2d = PSNR(ref_image, predicted_image_2d)
            cv2.imwrite(f'./out/2d_predicted_r{search_range}_b{macroblock_size}.jpg', predicted_image_2d)
            cv2.imwrite(f'./out/2d_motion_vector_r{search_range}_b{macroblock_size}.jpg', motion_vector_log_image)
            cv2.imwrite(f'./out/2d_residual_r{search_range}_b{macroblock_size}.jpg', residual_image_2d)
            print('2d_sad_value_r_{}_b_{}: {}'.format(search_range, macroblock_size, sad_2d))
            print('2d_psnr_r_{}_b_{}: {}'.format(search_range, macroblock_size, psnr_2d))
            print('2d_time_r_{}_b_{}: {} seconds'.format(search_range, macroblock_size, end-start))

    # Q2
    macroblock_size = 16
    search_range = 8
    predicted_image_full, motion_vector_full, residual_image_full = full_search(ref_image, target_image, macroblock_size, search_range)
    sad_full = SAD(test_image, predicted_image_full)
    psnr_full = PSNR(test_image, predicted_image_full)
    print('Q2_full_sad_value_r_{}_b_{}: {}'.format(search_range, macroblock_size, sad_full))
    print('Q2_full_psnr_r_{}_b_{}: {}'.format(search_range, macroblock_size, psnr_full))