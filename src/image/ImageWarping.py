import numpy as np
import os
from skimage import data
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt

import cv2

import skimage
import skimage.transform

from scipy.interpolate import Rbf



####Warping by PiecewiseAffine
def warp(image, src, dst):
    warp_trans = skimage.transform.PiecewiseAffineTransform()
    #warp_trans = skimage.transform.PolynomialTransform()
    warp_trans.estimate(dst, src)
    warped = skimage.transform.warp( image, warp_trans, output_shape=output_shape)
    warped = 255*warped                         # 0..1 => 0..255
    warped = warped.astype(np.uint8)            # convert from float64 to uint8
    return warped



####Warping by RBF
def shift_down(xy):
    xy[:, 1] -= 10
    return xy

class PointsRBF:
    def __init__(self, src, dst):
        xsrc = src[:,0]
        ysrc = src[:,1]
        xdst = dst[:,0]
        ydst = dst[:,1]
        # doc for RBF: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
        self.rbf_x = Rbf( xsrc, ysrc, xdst, function='multiquadric')  # learn the move for X: self.rbf_x learn to provide xdst from input=(xsrc,ysrc)
        self.rbf_y = Rbf( xsrc, ysrc, ydst, function='multiquadric')  # learn the move for Y: self.rbf_y learn to provide ydst from input=(xsrc,ysrc) 

    def __call__(self, xy):
        x = xy[:,0]
        y = xy[:,1]
        xdst = self.rbf_x(x,y)
        ydst = self.rbf_y(x,y)
        return np.transpose( [xdst,ydst] )


def warpRBF(image, src, dst):
    prbf = PointsRBF( dst, src)
    warped = skimage.transform.warp(image, prbf)
    warped = 255*warped                         # 0..1 => 0..255
    warped = warped.astype(np.uint8)            # convert from float64 to uint8
    return warped




def click(event, x, y, flags, param):
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(x,y)=",x," ",y, "   shape=", param[0].shape)
        a = np.array(  [ [x,y], [x,y],  ]).reshape( [2,1,2] )
        param[0] = np.append( param[0], a, 1)
        #a = np.append( param[0], [[x,y],], 0 )
        #b = np.append( param[1], [[x,y],], 0 )
        #param = np.array([a,b])
        #param.reshape( [2, param.shape[1]+1, 2] )
        #param[0] = a
        #param[1] = b
    elif event == cv2.EVENT_LBUTTONUP:
        param[0][1][-1] = [x,y]




if __name__ == '__main__':
    #image = data.coins()

    print(os.getcwd())
    image = data.load( os.getcwd() + "/data/neutral_face.jpg")

    output_shape = image.shape[:2]  # dimensions of our final image (from webcam eg)
    print("shape="+str(image.shape))
    src_coord = np.array([[0,0], [image.shape[1]-1,0], [image.shape[1]-1,image.shape[0]-1], [0,image.shape[0]-1], [186,331], [226,291], [264,327]])
    dst_coord = np.array([[0,0], [image.shape[1]-1,0], [image.shape[1]-1,image.shape[0]-1], [0,image.shape[0]-1], [170,320], [226,291], [285,320]])
    matching_array = np.array( [src_coord, dst_coord] )

    image_warped = warpRBF(image, src_coord, dst_coord)

    points = [ matching_array  ]
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click, points)

    while True:
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imw = cv2.cvtColor(image_warped, cv2.COLOR_BGR2RGB)
        matching_array = points[0]
        src_coord = matching_array[0]
        dst_coord = matching_array[1]
        for i in range(len(src_coord)):
            x = src_coord[i, 0]
            y = src_coord[i, 1]
            cv2.circle(im, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(imw, (x, y), 4, (0, 0, 255), -1)
            xd = dst_coord[i, 0]
            yd = dst_coord[i, 1]
            cv2.circle(im,  (xd, yd), 3, (255, 0, 0), -1)
            cv2.circle(imw, (xd, yd), 3, (255, 0, 0), -1)

            cv2.line( im,  (x, y), (xd,yd),  (255, 0, 0) )
            cv2.line( imw, (x, y), (xd,yd),  (255, 0, 0) )

        cv2.imshow("Frame", im)
        cv2.imshow("Frame2", imw)

        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key == 27:		# 27=ESC
            break
        if key == ord("w"):
            image_warped = warpRBF(image, matching_array[0], matching_array[1])

 
    # do a bit of cleanup
    cv2.destroyAllWindows()

