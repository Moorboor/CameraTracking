import MotionTracker as mt
import numpy as np
import cv2


def he(frame):

    def get_information_content(frame):
        hist = np.array([np.where(frame==i ,1 ,0).sum() for i in range(256)])
        prob = hist/hist.sum()
        info = np.where(prob>1e-6, np.log2(prob), 0)
        return info
    
    def get_entropy(frame):
        hist = np.array([np.where(frame==i ,1 ,0).sum() for i in range(256)])
        prob = hist/hist.sum()        
        entropy = -(prob * np.where(prob>1e-6, np.log2(prob), 0))[~np.isnan(prob)].sum()
        return entropy
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    hist = np.bincount(frame.flatten(), minlength=256)
    prob = hist/hist.sum()
    
    transfer = np.cumsum(prob) - np.arange(256)/256
    frame_he = (frame * (1+transfer[frame])).astype("uint8")
    return frame_he # , get_entropy(frame_he) - get_entropy(frame)




n = 11
kernel = np.ones(n**2).reshape(n,-1) * 1/n**2
kernel_sobel = np.array([-1,0,1] * 3).reshape(3,3)
kernel_sobel[1, :] *= 2
kernel_sobel = 1/4 * np.flip(kernel_sobel.T)

def convolve(frame):
    return cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

def erode(frame):
    return cv2.erode(frame, kernel=kernel)

def dilate(frame):
    return cv2.dilate(frame, kernel=kernel)



motionTrackerManager = mt.MotionTrackerManager(fig=False, lamb_bool=True, trajec=True)
motionTrackerManager.lamb_func = he# lambda x: (np.where(x>100, x, 0), (2))
motionTrackerManager.run()