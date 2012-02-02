import matplotlib.pyplot as plt
import Image
import scipy.ndimage
import numpy as np


if __name__ == '__main__':
    imgin = Image.open("/home/mh23g08/susanedge/susanedge/test_data/fish_image.jpg")
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point
    
    s = scipy.ndimage.sobel(img)
    
    plt.set_cmap(plt.cm.gray)
    plt.imshow(s)
    plt.show()
