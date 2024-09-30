
import os
import numpy as np
import cv2
import mediapipe as mp
import gc

from Vec3 import *

#mp_pose = mp.solutions.pose
#mp_pose = mp.solutions.pose
mp_pose_detector = mp.solutions.pose.Pose()     # en global pour éviter de le recréer à chaque fois dans la classe, sinon mettre en static


class Skeleton:
    """ class with a skeleton
        tab de Vec3

        # Full skeleton
        https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        0 - nose
        1 - left eye (inner)
        2 - left eye
        3 - left eye (outer)
        4 - right eye (inner)
        5 - right eye
        6 - right eye (outer)
        7 - left ear
        8 - right ear
        9 - mouth (left)
        10 - mouth (right)
        11 - left shoulder
        12 - right shoulder
        13 - left elbow
        14 - right elbow
        15 - left wrist
        16 - right wrist
        17 - left pinky
        18 - right pinky
        19 - left index
        20 - right index
        21 - left thumb
        22 - right thumb
        23 - left hip
        24 - right hip
        25 - left knee
        26 - right knee
        27 - left ankle
        28 - right ankle
        29 - left heel
        30 - right heel
        31 - left foot index
        32 - right foot index

        # Reduced skeleton
        ==> reduce 0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
        0 head
        1 left shoulder
        2 right shoulder
        3 left elbow
        4 right elbow
        5 left wrist
        6 right wrist
        7 left hip
        8 right hip
        9 left knee
        10 right knee
        11 left ankle
        12 right ankle
    """
    reduce_indice  = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    full_dim      = 33*3
    reduced_dim   = len(reduce_indice)*2

    colors_rgb = np.array([
                    [255, 0, 0],     # Rouge
                    [0, 255, 0],     # Vert
                    [0, 0, 255],     # Bleu
                    [255, 255, 0],   # Jaune
                    [255, 0, 255],   # Magenta
                    [0, 255, 255],   # Cyan
                    [128, 0, 0],     # Marron
                    [0, 128, 0],     # Vert foncé
                    [0, 0, 128],     # Bleu foncé
                    [255, 128, 0],   # Orange
                    [128, 0, 128],   # Pourpre
                    [128, 128, 0],   # Olive
                    [0, 128, 128],   # Sarcelle
                    [128, 128, 128], # Gris
                    [192, 192, 192], # Gris clair
                    [255, 165, 0],   # Or
                    [165, 42, 42],   # Brun
                    [0, 128, 192],   # Bleu acier
                    [128, 0, 128],   # Indigo
                    [128, 0, 0],     # Acajou
                    [128, 128, 0],   # Olive
                    [0, 128, 0],     # Vert foncé
                    [0, 128, 128],   # Sarcelle
                    [0, 0, 128],     # Bleu foncé
                    [0, 165, 255],   # Bleu royal
                    [165, 42, 42],   # Brun
                    [255, 140, 0],   # Orange rouge
                    [0, 250, 154],   # Vert clair
                    [75, 0, 130],    # Indigo foncé
                    [0, 255, 255],   # Cyan clair
                    [218, 112, 214], # Orchidée
                    [210, 105, 30],  # Chocolat
                    [240, 230, 140], # Kaki
                    [255, 20, 147],  # Rose profond
                ], dtype=np.uint8)


    def __init__(self):
        self.ske = np.empty( 33, dtype=Vec3)            # 33 is the size of mediapipe skeleton
        for i in range(33):
            self.ske[i] = Vec3(0,0,0)


    def __str__(self):          
        return str(self.ske)        

    
    def __array__(self, dtype=None, reduced=False):
        """ return skeleton as a numpy array of float, if reduced is True, keep only 13 minimals joints """
        if reduced:
            return np.vstack( self.ske[self.reduce_indice] ).astype(float)[:, :2]
        else:
            return np.vstack( self.ske ).astype(float)


    def reduce(self):
        return self.__array__(reduced=True)
    

    def fromImage(self, image):     
        """ get skeleton from image """
        #results = self.pose.process(image)
        results = mp_pose_detector.process(image)
        if results.pose_landmarks is None:
            return False
        if results.pose_landmarks:
            for index, landmark in enumerate(results.pose_landmarks.landmark):                    
                self.ske[index] = Vec3(landmark.x, landmark.y, landmark.z)
        ok = len(results.pose_landmarks.landmark)==33
        results.pose_landmarks.Clear()      # free memory of mo
        # del results
        # gc.collect()
        return ok


    def crop(self, x,y,w,h):
        """ crop skeleton """
        for i in range(33):
            self.ske[i].x = (self.ske[i].x - x) / w
            self.ske[i].y = (self.ske[i].y - y) / h


    def boundingBox(self):
        """ get bounding box of skeleton """
        minx, maxx = 1, 0
        miny, maxy = 1, 0
        for i in range(33):
            minx = min(minx, self.ske[i].x)
            maxx = max(maxx, self.ske[i].x)
            miny = min(miny, self.ske[i].y)
            maxy = max(maxy, self.ske[i].y)
        return minx, miny, maxx, maxy


    def distance(self, ske):        # TP-TODO
        """ distance between two skeletons """
        d = 0.0
        for i in range(33):
            d += norm( self.ske[i]-ske.ske[i])
        return d


    def draw(self, image):
        """ draw skeleton on image """
        image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height,width,_ = image.shape
        for i in range(33):
            x, y = int(self.ske[i].x * width), int(self.ske[i].y * height)
            cv2.circle(image, (x,y), 3, Skeleton.colors_rgb[i].tolist() , -1)
            #cv2.circle(image, (x,y), 3, (0, 0, 255), -1)
            #cv2.line(image, (100, 100), (500, 500), (0, 255, 0), 2)
        Skeleton.draw_reduced(self.reduce(), image)


    @staticmethod
    def neck(ske,w,h):
        ls = np.array( [int(ske[1][0] * w), int(ske[1][1] * h) ], dtype=int )
        rs = np.array( [int(ske[2][0] * w), int(ske[2][1] * h) ], dtype=int )
        return (0.5*(ls+rs)).astype(int)

    @staticmethod
    def pelvis(ske,w,h):
        lh = np.array( [ ske[7][0] * w, ske[7][1] * h ], dtype=int )
        rh = np.array( [ ske[8][0] * w, ske[8][1] * h ], dtype=int )
        return (0.5*(lh+rh)).astype(int)

    @staticmethod
    def joint(ske,w,h,idx):
        return np.array( [ ske[idx][0] * w, ske[idx][1] * h ], dtype=int )

    @staticmethod
    def color(idx):
        return ( int(Skeleton.colors_rgb[idx][0]),  int(Skeleton.colors_rgb[idx][1]),  int(Skeleton.colors_rgb[idx][2]) )

    @staticmethod
    def draw_reduced(skr, image):
        """ draw reduced skeleton on image 
                # 0 head
                # 1 left shoulder
                # 2 right shoulder
                # 3 left elbow
                # 4 right elbow
                # 5 left wrist
                # 6 right wrist
                # 7 left hip
                # 8 right hip
                # 9 left knee
                # 10 right knee
                # 11 left ankle
                # 12 right ankle
        """
        image.flags.writeable = True
        h,w,_ = image.shape
        pelvis = tuple(Skeleton.pelvis(skr,w,h))
        neck   = tuple(Skeleton.neck(skr,w,h))
        cv2.line(image, pelvis, neck, Skeleton.color(0), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,3), Skeleton.joint(skr,w,h,1), Skeleton.color(1), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,5), Skeleton.joint(skr,w,h,3), Skeleton.color(2), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,2), Skeleton.joint(skr,w,h,4), Skeleton.color(3), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,4), Skeleton.joint(skr,w,h,6), Skeleton.color(4), 4)
        cv2.line(image, neck, Skeleton.joint(skr,w,h,1), Skeleton.color(5), 4)
        cv2.line(image, neck, Skeleton.joint(skr,w,h,2), Skeleton.color(6), 4)
        cv2.line(image, neck, Skeleton.joint(skr,w,h,0), Skeleton.color(7), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,7), pelvis, Skeleton.color(8), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,8), pelvis, Skeleton.color(9), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,8), Skeleton.joint(skr,w,h,10), Skeleton.color(10), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,10), Skeleton.joint(skr,w,h,12), Skeleton.color(11), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,7), Skeleton.joint(skr,w,h,9), Skeleton.color(12), 4)
        cv2.line(image, Skeleton.joint(skr,w,h,9), Skeleton.joint(skr,w,h,11), Skeleton.color(13), 4)



if __name__ == '__main__':
    s = Skeleton()
    print("Current Working Directory:", os.getcwd())
    image = cv2.imread("tp/dance/test.jpg")
    if image is None:
        print('Lecture de l\'image a échoué.')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    s.fromImage(image)
    #print(s)
    print( "landmarks:", s )
    print( "landmarks as np:", s.__array__() )
    print( "landmarks as np:", s.__array__(reduced=True) )

    s.draw(image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()