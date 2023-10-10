import numpy as np
import cv2
import os
import pickle
import sys

from VideoReader import VideoReader
from Skeleton import Skeleton


def filename_change_ext(filename_full, nouvelle_extension):
    # Divise le nom de fichier en base et extension
    base = os.path.basename(filename_full)
    base = os.path.splitext(base)
    base = base[0]
    _, extension_actuelle = os.path.splitext(filename_full)
    path = os.path.dirname(filename_full)
    nouveau_nom_fichier = path + nouvelle_extension
    return nouveau_nom_fichier, path, base


def crop_image(image, x, y, width, height):
    """
    Crop an image and return the cropped region as a new image.
    Returns: numpy.ndarray: Cropped image.
    """
    # Crop the image using array slicing
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image



class VideoSkeleton:
    """ class that associate a skeleton to each frame of a video
    self.im : nparray<str> => im[i] filename of the image
    self.ske : nparray<Skeleton> => ske[i] the skeleton
    Proc draw() : display all the frame image+skeleton
    Fonc generator(Skeleton)->Image
     """
    def __init__(self, filename, forcecompute=False):
        new_video_width = 250           # video is resized to this width (the height is computed with the ratio)
        ske_width = 100                 # skeleton image is resized to this width
        ske_height = 150                # skeleton image is resized to this height        
        mod_frame = 350                 # only one frame every mod_frame is processed (taichi1.mp4 has 14000 frames with 20=>700 frames))

        filename_pkl, filename_dir, filename_base = filename_change_ext(filename, ".pkl")
        if not os.path.exists(filename_dir):
            os.makedirs(filename_dir)
        
        self.path = os.path.dirname(filename)
        if os.path.exists(filename_pkl) and os.path.exists(filename_dir) and not forcecompute:
            vs = VideoSkeleton.load(filename_pkl)
            self.ske = vs.ske
            self.im = vs.im
            return
        video = VideoReader(filename)
        print("read: "+filename+ " #frame="+str(video.getTotalFrames()))
        self.ske = np.empty( 0, dtype=Skeleton)
        self.im = []
        for i in range(video.getTotalFrames()):
            image = video.readFrame()
            #image_shape = image.shape
            if (i%mod_frame == 0):
                # resize full image
                ske = Skeleton()
                new_video_height = int(image.shape[0] * new_video_width / image.shape[1])
                image = cv2.resize(image, (new_video_width, new_video_height))
                if ske.fromImage( image ):      # a skeleton is found
                    filename_im = filename_base + "/image" + str(i) + ".jpg"
                    self.im.append(filename_im)
                    # crop
                    xm, ym, xM, yM = ske.boundingBox()
                    center_x = new_video_width * (xm + xM) / 2
                    center_y = new_video_height * (ym + yM) / 2
                    #print("frame "+str(i)+"/"+str(video.getTotalFrames())+" xm="+str(xm)+" ym="+str(ym)+" xM="+str(xM)+" yM="+str(yM))
                    xm = int(center_x-ske_width/2)
                    xM = int(center_x+ske_width/2)
                    ym = int(center_y-ske_height/2)
                    yM = int(center_y+ske_height/2)
                    image = image[ ym:yM, xm:xM ]           # image crop
                    ske.crop(xm/new_video_width, ym/new_video_height, ske_width/new_video_width, ske_height/new_video_height)   # skeleton crop
                    filename_imsave = filename_dir + "/" + filename_im
                    cv2.imwrite(filename_imsave, image)
                    self.ske = np.append(self.ske, ske)
                    print("frame "+str(i)+"/"+str(video.getTotalFrames()) + "   filename="+filename_im + "  save="+filename_imsave)
            cv2.destroyAllWindows()
        video.release()
        #self.im = self.im.reshape((self.ske.shape[0],) + image_shape)
        self.im = np.array(self.im)
        print("#skeleton="+str(self.ske.shape) + " #image="+str(self.im.shape))
        self.save( filename_pkl )


    def save(self,filename):
        with open(filename, "wb") as fichier:
            pickle.dump(self, fichier)
        print("save: "+filename)


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fichier:
            objet_charge = pickle.load(fichier)
        print("load: "+filename)
        return objet_charge


    def __str__(self):
        return str("VideoSkeleton: nbframe="+str(self.ske.shape))
    

    def draw(self):
        """ draw skeleton on image """
        print(os.getcwd())
        for i in range(self.ske.shape[0]):
            filename_im = self.path + "/" + self.im[i]
            im = cv2.imread(filename_im)
            # print(filename_im)
            self.ske[i].draw(im)
            cv2.imshow('Image', im)
            if cv2.waitKey(250) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        


if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "src/dance/data/taichi1.mp4"
    print("Current Working Directory:", os.getcwd())
    print("Filename=", filename)

    s = VideoSkeleton(filename, force)
    print(s)
    s.draw()

    # s.draw(image)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
