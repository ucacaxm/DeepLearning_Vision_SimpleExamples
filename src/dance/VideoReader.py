import cv2
import os

class VideoReader:
    """ Helper class for video utilities """
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0

    def readFrame(self):
        """ Read a frame """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is False or frame is None:
                return None
            self._current_frame += 1
        else:
            return None
        return frame

    def readNFrames(self, num_frames=1):
        """ Read n frames """
        frames_list = []
        for _ in range(num_frames):
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret is False or frame is None:
                    return None
                frames_list.append(frame)
                self._current_frame += 1
            else:
                return None
        return frames_list

    def isOpened(self):
        """ Check is video capture is opened """
        return self.cap.isOpened()

    def getFrameWidth(self):
        """ Get width of a frame """
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def getFrameHeight(self):
        """ Get height of a frame """
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def getVideoFps(self):
        """ Get Frames per second of video """
        return self.cap.get(cv2.CAP_PROP_FPS)

    def getCurrentFrame(self):
        """ Get current frame of video being read """
        return self._current_frame

    def getTotalFrames(self):
        """ Get total frames of a video """
        return self._total_frames

    def release(self):
        """ Release video capture """
        self.cap.release()

    def __del__(self):
        self.release()


if __name__ == '__main__':
    print("Current Working Directory:", os.getcwd())
    video = VideoReader("src/dance/data/taichi1.mp4")
    for i in range(video.getTotalFrames()):
        image = video.readFrame()
        cv2.imshow('Image', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()