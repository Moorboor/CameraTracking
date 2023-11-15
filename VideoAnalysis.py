import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime, timedelta




utc_time = datetime.utcnow()
date_string = utc_time.strftime('%Y-%m-%d')
time_string = utc_time.strftime('%H-%M-%S')
germany_offset = timedelta(hours=1)

# Create folder names
year_folder = utc_time.strftime('%Y')
month_folder = utc_time.strftime('%m')
day_folder = utc_time.strftime('%d')
folder_path = os.path.join(year_folder, month_folder, day_folder)


# Full path to save the file
ABS_PATH = os.path.abspath("")
RECORDING_PATH = os.path.join(ABS_PATH, "Recordings")
TODAY_RECORDING_PATH = os.path.join(folder_path, f'{time_string}.mp4')
os.makedirs(folder_path, exist_ok=True)
os.makedirs(RECORDING_PATH, exist_ok=True)



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 320*3) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 240*3) 



def set_focus(*, frame, height=100, width=80):
    return frame[height:, width::]

def render_text(*, frame, text=0.):
    text = f"{text:.0f}"
    position = (30, 30)  # (x, y) coordinates
    # Define the font and text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # BGR color (green in this example)
    font_thickness = 2
    cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
    return True

def save_video(*, recording):

    utc_time = datetime.utcnow()
    germany_time = utc_time + germany_offset
    time_string = germany_time.strftime('%H-%M-%S')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(os.path.join(folder_path, f"{time_string}.mp4"), fourcc, 20.0, (800, 600), True)  
    for f in recording: 
        out.write(np.array(f, dtype="uint8"))
    out.release()
    return True


frames = []
errors = []
recording_buffer = []
i = 0
n_saved = 0


kernels = [np.ones((5,5))/25 for _ in range(2)]
kernels.append(np.ones((10,10))/100)

while True:

    ret, frame_source = cap.read()
    if not ret:
        break

    cv2.imshow('Source', frame_source)
    frame = set_focus(frame=frame_source)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Preprocess', frame)
 

    frames.append(frame)
    if len(frames) > 2: del frames[0]
    recording_buffer.append(frame_source)
    if len(recording_buffer) > 20: del recording_buffer[0]
    

    mean_frame1 = np.array(frames[:1])
    mean_frame2 = np.array(frames[1:])
    error_frame = mean_frame1.mean(axis=0) - mean_frame2.mean(axis=0)
    error_frame = cv2.filter2D(src=error_frame, ddepth=-1, kernel=kernels[-1]) 
    error_frame = np.where(error_frame<8, 0, error_frame)


    error = (error_frame**2).mean()
    errors.append(error)
    render_text(frame=error_frame, text=error)
    cv2.imshow('Movement', error_frame)

    if error > 20 or i!=0:
        mean_error = np.array(errors[-10:]).mean()
        i += 1
        if i == 1:
            recording = recording_buffer.copy()
        recording.append(frame_source)

        if i > 80 and mean_error<3:
            i = 0
            save_video(recording=recording)
            n_saved += 1
            print(f"No.{n_saved} Video Saved!")

    # Press 'q' to exit the video playback
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


    
# Release the VideoCapture object
cap.release()


# Close all OpenCV windows
cv2.destroyAllWindows()
plt.plot(np.convolve(np.ones(20)/20, errors))
plt.show()