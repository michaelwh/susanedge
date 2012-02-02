from __future__ import division
import matplotlib.pyplot as plt
import Image
import scipy.ndimage
import numpy as np



# taken from ---------- http://stackoverflow.com/questions/2770356/extract-points-within-a-shape-from-a-raster
def intceil(x):
    return int(np.ceil(x))
        
def points_in_circle(circle, arr):
    "A generator to return all points whose indices are within given circle."
    i0,j0,r = circle
    circle_p = []
    points_x = []
    points_y = []
    
    for i in xrange(intceil(i0-r),intceil(i0+r)):
        ri = np.sqrt(r**2-(i-i0)**2)
        for j in xrange(intceil(j0-ri),intceil(j0+ri)):
            circle_p.append(arr[i][j])
            points_x.append(i)
            points_y.append(j)
    return (np.asarray(points_x), np.asarray(points_y), np.asarray(circle_p))
# end taken from ---------- http://stackoverflow.com/questions/2770356/extract-points-within-a-shape-from-a-raster

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

def simple_susan_fast(img, thresh=27.0, nmax=9.0):
    area = np.zeros_like(img)
    global_thresh = (3.0 * nmax) / 4.0
    for x in range(1, len(img) - 1):
        print x
        for y in range(1, len(img[x]) - 1):
            mask = img[x-1:x+2, y-1:y+2]
            mask = np.absolute(mask - img[x, y])
            mask = np.ma.masked_greater_equal(mask, thresh)
            a = mask.count()
            if a < global_thresh:
                area[x, y] = global_thresh - a
            else:
                area[x, y] = 0
    
    return area
    
def smooth_susan(img, thresh=27.0, nmax=9.0):
    area = np.zeros_like(img)
    global_thresh = (3.0 * nmax) / 4.0
    for x in range(1, len(img) - 1):
        print x
        for y in range(1, len(img[x]) - 1):
            mask = img[x-1:x+2, y-1:y+2]
            mask = np.absolute(mask - img[x, y])
            mask = np.exp(-np.power(mask/thresh, 6))
            a = mask.sum()
            if a < global_thresh:
                area[x, y] = global_thresh - a
            else:
                area[x, y] = 0
    
    return area

def step_edge_dir(usan_points_x, usan_points_y, usan, usan_sum, usan_x, usan_y):    
    centre_x = (usan_points_x*usan).sum() / usan_sum
    centre_y = (usan_points_y*usan).sum() / usan_sum
    
    edge_angle = np.tan((centre_y - usan_y) / (centre_x - usan_x)) + (np.pi / 2)
    
    if edge_angle == np.nan:
        edge_angle = 0
        
    edge_y = usan_sum*np.cos(edge_angle)
    edge_x = usan_sum*np.sin(edge_angle)
    
    return edge_angle, edge_x, edge_y

def smooth_susan_circle(img, thresh=27.0, r=3.4):
    area = np.zeros_like(img)
    directions_x = np.zeros_like(img)
    directions_y = np.zeros_like(img)
    angles = np.zeros_like(img)
    for x in range(intceil(r), len(img) - intceil(r)):
        print x
        for y in range(intceil(r), len(img[x]) - intceil(r)):
            points_x, points_y, mask = points_in_circle((x, y, r), img)
            usan = np.absolute(mask - img[x, y])
            usan = np.exp(-np.power(usan/thresh, 6))
            a = usan.sum()
            nmax = len(usan)
            global_thresh = (3.0 * nmax) / 4.0
            if a < global_thresh:
                area[x, y] = global_thresh - a
            else:
                area[x, y] = 0
            
            edge_angle, edge_x, edge_y = step_edge_dir(points_x, points_y, usan, a, x, y)
            
            angles[x, y] = edge_angle            
            directions_x[x, y] = edge_x
            directions_y[x, y] = edge_y
            
    return area, directions_x, directions_y, angles

if __name__ == '__main__':
    imgin = Image.open("/home/mh23g08/susanedge/susanedge/test_data/small_simple.png")
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point
    
    area_img, directions_x, directions_y, angles = smooth_susan_circle(img)   
    
    
    #angles_n =  (angles + angles.min()) * (255.0/angles.max())
    
    plt.set_cmap(plt.cm.gray)
    
    plt.quiver(directions_x, directions_y, color='r')
    plt.imshow(area_img)
    plt.show()
