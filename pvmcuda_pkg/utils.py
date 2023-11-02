# GPU PVM Implementation
# (C) 2017 Filip Piekniewski All Rights Reserved
# filip@piekniewski.info
import cv2
import os
import numpy as np
import errno
import logging


def compress_tensor(tensor, lower=0.1, higher=0.9):
    mul = higher - lower
    return tensor*mul + lower



def uncompress_tensor(tensor, lower=0.1, higher=0.9):
    mul = 1.0 /(higher - lower)
    return np.minimum(np.maximum((tensor-lower)*mul, 0), 1)


def check_if_enabled(switch, config_dic):
    if switch in list(config_dic.keys()):
        if int(config_dic[switch]) > 0:
            return True
        else:
            return False
    return False

def load_video(path, dx=None, dy=None, length=100000):
    cap = cv2.VideoCapture(os.path.expanduser(path))
    while not cap.isOpened():
        cap = cv2.VideoCapture(path)
        cv2.waitKey(1000)
        print("Wait for the header")
    frames = []
    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            if dx is not None and dy is not None:
                framer = cv2.resize(frame, dsize=(dx, dy), interpolation=cv2.INTER_CUBIC)
                framer = cv2.flip(framer, 0)
                framer = framer.astype(np.float32)/255
            else:
                framer = frame
                framer = cv2.flip(framer, 0)
            frames.append(framer)
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT) or len(frames) == length:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
    return frames

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class VideoRecorder(object):
    def __init__(self, rec_filename, fps=20):
        """
        :param rec_filename:
        :return:

        Handy object to carry out video recording
        """
        self.rec_filename = rec_filename
        self._video = None
        self.fps = fps

    def record(self, image):
        """
        :param image:
        :return:

        Takes and image and records it into a file
        """
        if self._video is None:
                self._video = cv2.VideoWriter()
                fps = self.fps
                retval = self._video.open(os.path.expanduser(self.rec_filename),
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                          fps, (image.shape[1], image.shape[0]))
                assert(retval)
                logging.info("Creating an avi file %s" % os.path.expanduser(self.rec_filename))
        self._video.write(image)

    def finish(self):
        """
        When done releases the cv video writer
        :return:
        """
        if self._video is not None:
            self._video.release()
            self._video = None
            logging.info("Finished recording file %s" % os.path.expanduser(self.rec_filename))