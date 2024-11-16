import MotionTracker as mt
import numpy as np
import cv2


def histogram_equalization(frame):

    def get_information_content(frame):
        hist = np.histogram(frame, bins=256)[0]
        prob = hist/hist.sum()
        info = np.where(prob>1e-6, np.log2(prob), 0)
        return info
    
    def get_entropy(frame):
        info = get_information_content(frame)
        hist = np.histogram(frame, bins=256)[0]
        prob = hist/hist.sum()
        
        entropy = -(prob * info)[~np.isnan(prob)].sum()
        return entropy
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = np.array([np.where(frame==i ,1 ,0).sum() for i in range(256)])
    prob = hist/hist.sum()
    transfer = np.cumsum(prob) - np.arange(256)/256
    frame_he = (frame * (1+transfer[frame])).astype("uint8")
    return frame_he, get_entropy(frame_he) - get_entropy(frame)


motionTrackerManager = mt.MotionTrackerManager(fig=True, lamb_bool=True)
motionTrackerManager.lamb_func = histogram_equalization# lambda x: (np.where(x>100, x, 0), (2))
motionTrackerManager.run()