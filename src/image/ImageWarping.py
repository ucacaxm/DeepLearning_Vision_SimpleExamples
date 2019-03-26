import numpy as np
import os
from skimage import data
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt

import cv2

import skimage
import skimage.transform




if __name__ == '__main__':
    #image = data.coins()

    print(os.getcwd())
    image = data.load( os.getcwd() + "/data/neutral_face.jpg")

    output_shape = image.shape[:2]  # dimensions of our final image (from webcam eg)
    print("shape="+str(image.shape))
    src_coord = np.array([[0,0], [image.shape[1]-1,0], [image.shape[1]-1,image.shape[0]-1], [0,image.shape[0]-1], [150,200], [100,100], [150,100]])
    dst_coord = np.array([[0,0], [image.shape[1]-1,0], [image.shape[1]-1,image.shape[0]-1], [0,image.shape[0]-1], [190,200], [100,100], [150,80]])
    warp_trans = skimage.transform.PiecewiseAffineTransform()
    #warp_trans = skimage.transform.PolynomialTransform()
    warp_trans.estimate(dst_coord, src_coord)
    image_warped = skimage.transform.warp( image, warp_trans, output_shape=output_shape)
    image_warped = 255*image_warped                         # 0..1 => 0..255
    image_warped = image_warped.astype(np.uint8)            # convert from float64 to uint8

    while True:
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imw = cv2.cvtColor(image_warped, cv2.COLOR_BGR2RGB)
        for i in range(len(src_coord)):
            x = src_coord[i, 0]
            y = src_coord[i, 1]
            cv2.circle(im, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(imw, (x, y), 4, (0, 0, 255), -1)
            x = dst_coord[i, 0]
            y = dst_coord[i, 1]
            cv2.circle(im, (x, y), 3, (255, 0, 0), -1)
            cv2.circle(imw, (x, y), 3, (255, 0, 0), -1)

        cv2.imshow("Frame", im)
        cv2.imshow("Frame2", imw)

        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key == 27:		# 27=ESC
            break
 
    # do a bit of cleanup
    cv2.destroyAllWindows()

