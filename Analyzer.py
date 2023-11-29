import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt
from VideoAnalysis import set_focus
import pandas as pd

ABS_PATH = os.path.abspath("")
CSV_PATH = os.path.join(ABS_PATH, f"CSV")
os.makedirs(CSV_PATH, exist_ok=True)


paths = {}
dates = []

for (root, _, files) in os.walk(os.path.join(ABS_PATH, "2023", "11", "26")):
    if files != []: 
        paths[root] = files
        date = root.split("CameraTracking")
        date = date[-1].replace("\\", "-")[1:]
        dates.append(date)




# ------ #
# Analyze videos from folders and label them into three categories
# CSV listing all fnames and error values incorporating error functions

def write_csv(*, df, y):
    df.to_csv(os.path.join(CSV_PATH, dates[y]), index=False)
    return True

def analyze_video(*, fname):

    
    cap = cv2.VideoCapture(fname)
    frame_source = []
    ret = True
    
    while(ret):
        ret, frame = cap.read()
        if not ret:
            break
        frame_source.append(frame)
    frames = []
    errors = []

    for frame in frame_source:
        frame = set_focus(frame=frame, start_height=100, start_width=80)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        if len(frames) > 2: del frames[0]
        mean_frame1 = np.array(frames[:1])
        mean_frame2 = np.array(frames[1:])
        error_frame = mean_frame1.mean(axis=0) - mean_frame2.mean(axis=0)
        error_frame = cv2.filter2D(src=error_frame, ddepth=-1, kernel=kernel) 
        error_frame = np.where(error_frame<5, 0, error_frame)
        error = (error_frame**2).mean()

        errors.append(error)
    return np.nanmax(np.convolve(np.ones(10)/10, np.array(errors)))


kernel = np.ones((10,10))/100


df = pd.DataFrame(columns = ['Date', 'Error', 'Label'])
for y, (key, value) in enumerate(paths.items()):
    for i, fname in enumerate(paths[key]):
        error = analyze_video(fname=os.path.join(key, fname))

        if error >= 100:
            label = "Bus"
        elif 10 < error < 100:
            label = "Car"
        else: label = "Passenger"
        df.loc[i] = [fname.replace(".mp4", ""), f"{error:.2f}", label]

    write_csv(df=df,y=y)
