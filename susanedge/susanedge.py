from __future__ import division
import matplotlib.pyplot as plt
import Image
import scipy.ndimage as ndimage
import numpy as np
import sys
from optparse import OptionParser

_circle_points_x = np.array([0, 0, 0, # left
                            2, 3, 4, # top
                            2, 3, 4, # bottom
                            6, 6, 6, # right
                            1, 2, 3, 4, 5,
                            1, 2, 3, 4, 5,
                            1, 2, 3, 4, 5,
                            1, 2, 3, 4, 5,
                            1, 2, 3, 4, 5])

_circle_points_y = np.array([2, 3, 4, # left
                      0, 0, 0, # top
                      6, 6, 6, # bottom
                      2, 3, 4, # right
                      1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5])            

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

def points_in_circle_fast(circle, arr):
    "A generator to return all points whose indices are within given circle."
    i0,j0 = circle
    points_x = _circle_points_x + i0
    points_y = _circle_points_y + j0
    
    circle_p = arr[i0-3, j0-1:j0+2].reshape(-1) # left
    circle_p = np.append(circle_p, arr[i0-1:i0+2, j0-3].reshape(-1)) # top
    circle_p = np.append(circle_p, arr[i0-1:i0+2, j0+3].reshape(-1)) # top
    circle_p = np.append(circle_p, arr[i0+3, j0-1:j0+2].reshape(-1)) # right
    circle_p = np.append(circle_p, arr[i0-2:i0+3, j0-2:j0+3].reshape(-1)) # centre
    return (points_x, points_y, circle_p)

def simple_susan(img, thresh=27.0, nmax=9.0):
    area = np.zeros_like(img)
    global_thresh = (3.0 * nmax) / 4.0
    for x in range(1, len(img) - 1):
        print x
        for y in range(1, len(img[x]) - 1):
            a = 0
            for x1 in range(x-1, x+2):
                for y1 in range(y-1, y+2):
                    if abs(img[x1, y1] - img[x, y]) < thresh:
                        a += 1
            if a < global_thresh:
                area[x, y] = global_thresh - a
            else:
                area[x, y] = 0
    return area

def simple_susan_fast(img, thresh=27.0, nmax=9.0):
    area = np.zeros_like(img)
    global_thresh = (3.0 * nmax) / 4.0
    for x in range(1, len(img) - 1):
        print x
        for y in range(1, len(img[x]) - 1):
            mask = img[x-1:x+2, y-1:y+2]
            mask = np.absolute(mask - img[x, y])
            #mask = np.ma.masked_greater_equal(mask, thresh)
            mask_greater = mask > thresh
            a = mask_greater.sum()
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


def step_edge_dir(centre_x, centre_y, usan_points_x, usan_points_y, usan, usan_sum, usan_x, usan_y):    

    #cx = centre_x - usan_x
    #if cx == 0:
    #    edge_angle = 0
    #else:
    #edge_angle = np.tan((centre_y - usan_y) / cx)
        
    diff = np.array([centre_x - usan_x, centre_y - usan_y])
    diff_n = diff/np.sqrt(diff**2)
    edge_angle = np.arctan(diff_n[1] / diff_n[0])
    
    #edge_y = np.cos(edge_angle)
    #edge_x = np.sin(edge_angle)
    
    return edge_angle, diff_n[0], diff_n[1]
    
def band_edge_dir(usan_points_x, usan_points_y, usan, usan_sum, usan_x, usan_y):
    x_val = (np.power((usan_points_x - usan_x), 2) * usan).sum()
    y_val = (np.power((usan_points_y - usan_y), 2) * usan).sum()
    both_val = (((usan_points_x - usan_x) * (usan_points_y - usan_y)) * usan).sum()
    edge_angle = np.absolute(y_val / x_val) + np.pi/2
    if both_val < 0:
        edge_angle = -edge_angle
    
    #edge_angle += np.pi/2
    
    edge_y = np.cos(edge_angle)
    edge_x = np.sin(edge_angle)
    
    return edge_angle, edge_x, edge_y

def find_edge_dir(usan_points_x, usan_points_y, usan, usan_sum, usan_x, usan_y, r, thresh):
    centre_x = (usan_points_x*usan).sum() / usan_sum
    centre_y = (usan_points_y*usan).sum() / usan_sum
    
    #print centre_x, centre_y, usan_x, usan_y
    
    nuc_dist = np.sqrt((centre_x - usan_x)**2 + (centre_y - usan_y)**2)
    
    if (usan > thresh).sum() < (2*r) and nuc_dist > 1.0:
        #return (2, 1, 2)
        return band_edge_dir(usan_points_x, usan_points_y, usan, usan_sum, usan_x, usan_y)
    else:
        #return (0, 0, 0)
        return step_edge_dir(centre_x, centre_y, usan_points_x, usan_points_y, usan, usan_sum, usan_x, usan_y)
    

# begin code from http://old.nabble.com/canny1d-filter--td23223688.html ----------------------    
_N  = np.array([[0, 1, 0], 
                   [0, 0, 0], 
                   [0, 1, 0]], dtype=bool) 

_NE = np.array([[0, 0, 1], 
                   [0, 0, 0], 
                   [1, 0, 0]], dtype=bool) 

_W  = np.array([[0, 0, 0], 
                   [1, 0, 1], 
                   [0, 0, 0]], dtype=bool) 

_NW = np.array([[1, 0, 0], 
                   [0, 0, 0], 
                   [0, 0, 1]], dtype=bool) 

def nonmax_supress(area_img, angles):
    _NE_d = 0
    _W_d = 1
    _NW_d = 2 
    _N_d = 3
    quantized_angle = np.around(angles) % 4
    NE = ndimage.maximum_filter(area_img, footprint=_NE) 
    W  = ndimage.maximum_filter(area_img, footprint=_W) 
    NW = ndimage.maximum_filter(area_img, footprint=_NW) 
    N  = ndimage.maximum_filter(area_img, footprint=_N) 
    thinned = (((area_img >= W)  & (quantized_angle == _N_d )) | 
             ((area_img >= N)  & (quantized_angle == _W_d )) | 
             ((area_img >= NW) & (quantized_angle == _NE_d)) | 
             ((area_img >= NE) & (quantized_angle == _NW_d)) ) 
    thinned_grad = thinned * area_img
    return thinned_grad
    #return (area_img > W)
# end code from http://old.nabble.com/canny1d-filter--td23223688.html ----------------------    
    
def smooth_susan_circle(img, thresh=27.0, r=3.4):
    area = np.zeros_like(img)
    directions_x = np.zeros_like(img)
    directions_y = np.zeros_like(img)
    angles = np.zeros_like(img)
    for x in range(intceil(r), len(img) - intceil(r)):
        print x
        for y in range(intceil(r), len(img[x]) - intceil(r)):
            # http://www.bmva.org/bmvc/1992/bmvc-92-015.pdf implement the circle talked about here, with a 5x5 square with 3 pixels added on each side
            #points_x, points_y, mask = points_in_circle((x, y, r), img)
            points_x, points_y, mask = points_in_circle_fast((x, y), img)
            usan = np.absolute(mask - img[x, y])
            usan = np.exp(-np.power(usan/thresh, 6))
            a = usan.sum()
            nmax = len(usan)
            global_thresh = (3.0 * nmax) / 4.0
            if a < global_thresh:
                area[x, y] = global_thresh - a
            else:
                area[x, y] = 0
            
            edge_angle, edge_x, edge_y = find_edge_dir(points_x, points_y, usan, a, x+3, y+3, r, thresh)
                
            angles[x, y] = edge_angle            
            directions_x[x, y] = edge_x
            directions_y[x, y] = edge_y
            
    return area, directions_x, directions_y, angles

def run_prog():
    parser = OptionParser()
    parser.add_option("-n", "--no-dir", help="don't plot direction on the image", action="store_true", default=False)
    parser.add_option("-m", "--no-nomax", help="don't perform nonmax supression on the image", action="store_true", default=False)
    
    (options, args) = parser.parse_args()

    #imgin = Image.open("/home/mh23g08/susanedge/susanedge/test_data/fish_image_small.jpg")
    imgin = Image.open(args[0])
	
    imgin = imgin.convert("L") # convert to greyscale (luminance)
    
    img = np.asarray(imgin)
    img = img.astype(np.float32) # convert to a floating point
    
    

    #area_img = simple_susan(img, thresh=10.0)
    area_img, directions_x, directions_y, angles = smooth_susan_circle(img, thresh=27.0)
    if not options.no_nomax:
        area_img = nonmax_supress(area_img, angles)
    #area_img = smooth_susan(img, thresh=10.0)
    #return None 
    #angles_n =  (angles + angles.min()) * (255.0/angles.max())
    plt.set_cmap(plt.cm.gray)
    if not options.no_dir:
        plt.quiver(directions_x, directions_y, color='r')
    plt.imshow(area_img)
    plt.show()

if __name__ == '__main__':
    #import timeit
    #t = timeit.Timer("run_prog()", "from __main__ import run_prog")
    
    #print t.timeit(1)
    run_prog()
    