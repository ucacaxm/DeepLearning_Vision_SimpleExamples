
import os
import numpy as np
import cv2
import mediapipe as mp

from Vec3 import *

#mp_pose = mp.solutions.pose
mp_pose = mp.solutions.pose



class Skeleton:
    """ class with a skeleton
        tab de Vec3
    """
    def __init__(self):
        self.ske = np.empty( 33, dtype=Vec3)            # 33 is the size of mediapipe skeleton
        for i in range(33):
            self.ske[i] = Vec3(0,0,0)


    def __str__(self):          
        return str(self.ske)        


    def fromImage(self, image):
        """ get skeleton from image """
        # image.flags.writeable = False
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # with mp_pose.Pose(
        #     min_detection_confidence=0.5,
        #     min_tracking_confidence=0.5) as pose:
        pose = mp_pose.Pose()
        results = pose.process(image)
        if results.pose_landmarks is None:
            return False
        # else:
        #     num_landmarks = len(results.pose_landmarks.landmark)
        #     print(f"Nombre d'articulations détectées : {num_landmarks}")
        #results = pose.process(image)
        if results.pose_landmarks:
            for index, landmark in enumerate(results.pose_landmarks.landmark):                    
                self.ske[index] = Vec3(landmark.x, landmark.y, landmark.z)
                #print( str(self.ske[index]) + " %.2f %.2f %.2f"% (landmark.x, landmark.y, landmark.z) )
        return len(results.pose_landmarks.landmark)==33


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


    def draw(self, image):
        """ draw skeleton on image """
        image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height,width,_ = image.shape
        for i in range(33):
            x, y = int(self.ske[i].x * width), int(self.ske[i].y * height)
            cv2.circle(image, (x,y), 3, (0, 0, 255), -1)
            #cv2.line(image, (100, 100), (500, 500), (0, 255, 0), 2)



if __name__ == '__main__':
    s = Skeleton()
    print("Current Working Directory:", os.getcwd())
    image = cv2.imread("src/dance/test.jpg")
    if image is None:
        print('Lecture de l\'image a échoué.')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    s.fromImage(image)
    #print(s)
    s.draw(image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()