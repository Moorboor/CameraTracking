import numpy as np
import os
import cv2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Germany's time zone offset from UTC is +1 hour
germany_offset = timedelta(hours=1)


ABS_PATH = os.path.abspath("")
RECORDING_PATH = os.path.join(ABS_PATH, "Recordings")
os.makedirs(RECORDING_PATH, exist_ok=True)



def save_video():

    germany_offset = timedelta(hours=1)
    utc_time = datetime.utcnow()
    germany_time = utc_time + germany_offset
    formatted_time = germany_time.strftime('%Y-%m-%d %H:%M:%S')
    
    return formatted_time


imgs = []
for i in range(100):
    img = np.linspace(0, i*np.pi**0.5, 800).reshape(1,800) *  np.linspace(0, i*np.pi**0.5, 800).reshape(800, 1)
    img = np.sin(img)*200

    imgs.append(np.array(img, dtype="uint8"))






fnames = [int(img.replace(".jpg", "")) for img in os.listdir(RECORDING_PATH)]
fnames = sorted(fnames)
r_imgs = [cv2.imread(os.path.join(RECORDING_PATH, f"{fname}.jpg")) for fname in fnames]



#out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (370, 370), True)  # 'output.mp4' is the output file name
for f in r_imgs:
    f = f[55:-55, 140:-130]
    print(f.shape)
    out.write(f)
out.release()
