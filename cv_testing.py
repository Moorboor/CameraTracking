import MotionTracker as mt
import numpy as np
import cv2


def histogram_equalize(frame):

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






element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
def erode(frame):
    return cv2.erode(frame, kernel=element)

def dilate(frame):
    return cv2.dilate(frame, kernel=element)

def hough_transform(frame):
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp = cv2.Canny(temp, 100, 150, apertureSize=3)

    thetas = np.linspace(-np.pi/2, np.pi/2, 180)
    hough_space = np.zeros_like(frame)
    p = np.where(temp)
    r = p[0][:, None]*np.cos(thetas) + p[1][:, None]*np.sin(thetas) 
    r = np.clip(r, -frame.shape[0], frame.shape[0]-1)


    for x in np.array_split(r, indices_or_sections=10):
        for i in range(len(thetas)):
            hough_space[x[:,i].astype(int), i*7:(i+1)*7] += 255//10
    lines = None
    # lines = cv2.HoughLines(temp, rho=1, theta=np.pi/180, threshold=250)
    if lines is not None:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    return hough_space
 


def skeletonize(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    size = np.size(frame)
    skel = np.zeros(frame.shape, dtype=np.uint8)
    
    ret, frame = cv2.threshold(frame, 127, 255, 0)
    
    while True:
        eroded = cv2.erode(frame, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(frame, temp)
        skel = cv2.bitwise_or(skel, temp)
        frame = eroded.copy()
    
        zeros = size - frame.sum()
        if zeros==size:
            break
    return skel

motionTrackerManager = mt.MotionTrackerManager(fig=False, lamb_bool=True, trajec=False, cam=True, mtracker=False)
motionTrackerManager.lamb_func = hough_transform
motionTrackerManager.run()