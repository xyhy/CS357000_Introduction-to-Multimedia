import numpy as np
import cv2
import matplotlib.pyplot as plt



##for(a)
def subplot(points, img):

    # low detail
    plt.imshow(img)
    
    plt.scatter(points[:, 0], points[:, 1],  s=0.5)
    
    for i in range(0, len(points)-1, 3):
        point = points[i:i + 4]
        [Bx, By] = cal_curve(point, 0.5)
        plt.plot(Bx, By, 'b-' ,linewidth=0.5)

    #high detail
    for i in range(0, len(points)-1, 3):
        point = points[i:i + 4]
        [Bx, By] = cal_curve(point, 0.01)
        plt.plot(Bx, By, 'r-' ,linewidth=0.5)
    plt.savefig('./output/1a.png')
    plt.close()

##for(b)
def plot(points , img):
    # print(img.shape)
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1],  s=5)
    for i in range(0, len(points)-1, 3):
        point = points[i:i + 4]
        [Bx, By] = cal_curve(point, 0.01)
        plt.plot(Bx, By, 'r-' ,linewidth=0.5)
    plt.savefig('./output/1b.png')
    plt.close()


def cal_curve(point, ratio):
    Bx = []
    By = []
    for t in np.arange(0, 1.01, ratio):
        x = ((1 - t) ** 3) * point[0, 0] + 3 * ((1 - t) ** 2) * t * point[1, 0] + 3 * (1 - t) * (t ** 2) * point[2, 0]+ (t ** 3) * point[3, 0]
        y = ((1 - t) ** 3) * point[0, 1] + 3 * ((1 - t) ** 2) * t * point[1, 1] + 3 * (1 - t) * (t ** 2) * point[2, 1]+ (t ** 3) * point[3, 1]
        Bx.append(x)
        By.append(y)

    return [Bx, By]

def nnScale(img):
    height, width = img.shape[0], img.shape[1]
    newh, neww = height*4, width*4
    scale_Img = np.zeros((newh, neww, 3), dtype=np.uint8)
    for y in range(newh):
        for x in range(neww):
            oriy = int(np.floor(y/4.0))
            orix = int(np.floor(x/4.0))
            scale_Img[y, x, :] = img[oriy, orix, :]
    return scale_Img

def main():      
    # Load the image and points
    img = cv2.imread("./bg.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = np.loadtxt("./points.txt")
    
    
    ##You shold modify result1 , result2 , result
    ## 1.a
    subplot(points, img)
    
    # 2.a 
    # cv2 nn scale
    # width = int(img.shape[1] * 4)
    # height = int(img.shape[0] * 4)
    # dim = (width, height)
    # scale_img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
    
    # self nn scale
    scale_img = nnScale(img)
    newpoints = points*4
    plot(newpoints, scale_img)
    

if __name__ == "__main__":
    main()