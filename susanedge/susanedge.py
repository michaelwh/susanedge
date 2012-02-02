import matplotlib.pyplot as plt
import Image
import scipy.ndimage
import numpy as np



def simple_susan(img, thresh=27.0):
    area = np.zeros_like(img)
    for x in range(1, len(img) - 1):
        print x
        for y in range(1, len(img[x]) - 1):
            area[x, y] = 0.0
            for x1 in range(x-1, x+2):
                for y1 in range(y-1, y+2):
                    if abs(img[x1, y1] - img[x, y]) < thresh:
                        area[x, y] += 1
    return area
        


if __name__ == '__main__':
    imgin = Image.open("/home/mh23g08/susanedge/susanedge/test_data/simple_shapes.png")
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point
    
    area_img = simple_susan(img)   
    
    plt.set_cmap(plt.cm.gray)
    plt.imshow(area_img)
    plt.show()
