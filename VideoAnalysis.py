import numpy as np
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
TODAY_RECORDING_PATH = os.path.join(folder_path, f'{time_string}.mp4')
os.makedirs(folder_path, exist_ok=True)



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 320*3) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 240*3) 



def set_focus(*, frame, start_height=0, height=0, start_width=0, width=0):
    if height == 0: height = frame.shape[0]
    if width == 0: width = frame.shape[1]
    return frame[start_height:height, start_width:width]

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
    out = cv2.VideoWriter(os.path.join(folder_path, f"{time_string}.mp4"), fourcc, 20.0, (recording[-1].shape[1], recording[-1].shape[0]), True)  
    for f in recording: 
        out.write(np.array(f, dtype="uint8"))
    out.release()
    return True

def new_glow_gradient(*, glow_gradient, new_frame):
    glow_gradient = np.array(glow_gradient, dtype="float")
    glow_gradient *= .9
    glow_gradient += np.array(new_frame, dtype="float")
    #glow_gradient = np.where(glow_gradient<5, 0, glow_gradient)
    return np.array(glow_gradient, dtype="uint8")

def add_glow(*, frame, glow_gradient):
    frame = set_focus(frame=frame, start_height=100, start_width=80)
    frame = np.array(frame, dtype="float")
    glow_gradient = np.array(glow_gradient, dtype="float")
    color_gradient = np.zeros_like(frame)
    color_gradient[:, :, 0] = 2*glow_gradient
    #color_gradient[:, :, 1] = glow_gradient/2 
    frame += color_gradient
    return np.array(frame, dtype="uint8")


def record_video(*, record_bool=True):

    frames = []
    errors = []
    recording_buffer = []
    glow_gradient = np.zeros(shape=(500, 720))
    i = 0
    n_saved = 0
    glow_bool = False

    kernel = (np.ones((10,10))/100)

    while True:

        ret, frame_source = cap.read()
        if not ret:
            break

        # cv2.imshow('Source', frame_source)
        frame = set_focus(frame=frame_source, start_height=100, start_width=80)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

        # Recording
        frames.append(frame)
        if len(frames) > 2: del frames[0]
        recording_buffer.append(frame_source)
        if len(recording_buffer) > 20: del recording_buffer[0]
        

        mean_frame1 = np.array(frames[:1])
        mean_frame2 = np.array(frames[1:])
        error_frame = mean_frame1.mean(axis=0) - mean_frame2.mean(axis=0)
        error_frame = cv2.filter2D(src=error_frame, ddepth=-1, kernel=kernel) 
        error_frame = np.where(error_frame<5, 0, error_frame)
        error = (error_frame**2).mean()
        errors.append(error)
        render_text(frame=error_frame, text=error)
        # cv2.imshow('Movement', np.array(error_frame, dtype="uint8"))
        
        if glow_bool:
            glow_gradient = new_glow_gradient(glow_gradient=glow_gradient, new_frame=error_frame)
        # cv2.imshow("Glow", glow_gradient)
        color_glow = add_glow(frame=frame_source, glow_gradient=glow_gradient)
        cv2.imshow("Filter", color_glow)


        if error > 5 or i != 0:
            mean_error = np.array(errors[-10:]).mean()
            i += 1
            if i == 1:
                recording = recording_buffer.copy()
            recording.append(frame_source)

            if i > 80 and mean_error < 3:
                i = 0
                if record_bool:
                    save_video(recording=recording)
                    n_saved += 1
                    print(f"No.{n_saved} video saved!")


        # Quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break 
        if cv2.waitKey(25) & 0xFF == ord('1'):
            glow_bool = not glow_bool
            print(f"Glow: {glow_bool}")
            
        if n_saved == 20:
            break

    # Release the VideoCapture object
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    return n_saved  

if __name__ == "__main__":
    n_saved = record_video(record_bool=True)

    print(f"Total {n_saved} videos saved!")