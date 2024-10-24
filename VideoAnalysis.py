import numpy as np
import os
import cv2
from datetime import datetime, timedelta
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk


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



# cap = cv2.VideoCapture(2) # for MacBook index 2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 800) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 600) 


cv2.namedWindow("Filter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Filter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def set_focus(*, frame, width_corr, height_corr):
    if height_corr[1] == 0: 
        height_corr[1] = frame.shape[0]
    if width_corr[1] == 0: 
        width_corr[1] = frame.shape[1]
    return frame[height_corr[0]:height_corr[1], width_corr[0]:width_corr[1]]

def render_text(*, frame, text=0., position=(30,30)):
    text = f"{text:.0f}"
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
    glow_gradient *= .5
    glow_gradient += np.array(new_frame, dtype="float")
    return np.array(glow_gradient, dtype="uint8")

def add_glow(*, frame, color_code, glow_gradient):
    frame = np.array(frame, dtype="float")
    glow_gradient = np.array(glow_gradient, dtype="float")
    color_gradient = np.zeros_like(frame)
    color_gradient[:, :, color_code] = 2*glow_gradient
    color_gradient[:,:, 0] += glow_gradient
    color_gradient[:,:, 1] += glow_gradient
    color_gradient[:,:, 2] += glow_gradient
    frame += color_gradient
    return np.array(color_gradient, dtype="uint8")

def get_error_frame(*, frames, kernel):
    mean_frame1 = np.array(frames[:1])
    mean_frame2 = np.array(frames[1:])
    error_frame = mean_frame1.mean(axis=0) - mean_frame2.mean(axis=0)
    error_frame = cv2.filter2D(src=error_frame, ddepth=-1, kernel=kernel) 
    error_frame = np.where(error_frame<5, 0, error_frame)
    error = (error_frame**2).mean()
    return error_frame, error


class custom_figure():
    
    def __init__(self):
        root = Tk()
        root.withdraw()  # Hide the dummy root window
        plt.rcParams['toolbar'] = 'None'
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(16,9))
        colors = sns.color_palette("rocket", 3)[1:]
        self.fig.patch.set_facecolor('#000000')
        self.ax1.set_facecolor('#000000')
        self.ax2.set_facecolor('#000000')


        # Remove bars
        plt.get_current_fig_manager().window.attributes('-alpha', 0.3)
        plt.get_current_fig_manager().window.overrideredirect(True)
        # Remove figure Name
        # plt.get_current_fig_manager().window.title("")

        (self.ln1,) = self.ax1.plot(range(100), np.linspace(0,150, 100), animated=True, c=colors[0])
        (self.ln2,) = self.ax2.plot(np.linspace(-50, 350, 100), np.linspace(0,0.35,100), animated=True, c=colors[1])
        self.ax2.fill_between(np.linspace(-50, 350, 100), 0, 0, color='blue', label="filling")

        plt.show(block=False)
        plt.pause(0.1)
        plt.get_current_fig_manager().window.state('zoomed')
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.ax1.draw_artist(self.ln1)
        self.ax2.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)


    def update_figure(self, *, errors):


        def central_moments(*, x, k):
            return ((x-x.mean())**k).mean()
        
        def get_distribution(*, x, mu, std):
            return (1/std*(2*np.pi)**0.5) * np.exp(-0.5*((x-mu)/std)**2)
        
        errors_dist = np.array(errors)[-25:]
        mean = errors_dist.mean()
        std = central_moments(x=errors_dist, k=2)**0.5
        # skewness = central_moments(x=errors_dist, k=3)/(central_moments(x=errors_dist, k=2)**(0.5*3))
        # kurtosis = central_moments(x=errors_dist, k=4)/(central_moments(x=errors_dist, k=2)**(0.5*(4)))
        dist = get_distribution(x=np.linspace(-50,350, 100),mu=mean, std=std)

        self.fig.canvas.restore_region(self.bg)
        
        self.ln1.set_ydata(errors)
        self.ln2.set_ydata(dist)

        # re-render the artist, updating the canvas state, but not the screen
        path = self.ax2.fill_between(np.linspace(-50, 350, 100), 0, dist, label="filling", alpha=0.8)
        self.ax2.draw_artist(path)
        self.ax1.draw_artist(self.ln1)
        self.ax2.draw_artist(self.ln2)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

        return True


def record_video(*, record_bool=True, plot_bool=True):


    frames = []
    errors = []
    kernel = (np.ones((10,10))/100)
    width_corr, height_corr = (0, 640), (0, 400)
    # width_corr, height_corr = (430, 480), (380, 390)

    
    glow_bool = False
    glow_gradient = np.zeros(shape=(height_corr[1]-height_corr[0], width_corr[1]-width_corr[0]))

    
    i = 0
    n_saved = 0
    recording_buffer = []

    
    start = time.time()


    if plot_bool == True: 
        figure = custom_figure()
    
    while True:


        ret, frame_source = cap.read()
        if not ret:
            print("Error: cap not found")
            break

        frame = set_focus(frame=frame_source, width_corr=width_corr, height_corr=height_corr)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  


        frames.append(frame)
        if len(frames) > 2: 
            del frames[0]
        error_frame, error = get_error_frame(frames=frames, kernel=kernel)
        errors.append(error)
        render_text(frame=frame_source, text=error)


        
        recording_buffer.append(frame_source)
        if len(recording_buffer) > 20: 
            del recording_buffer[0]


        if plot_bool and len(errors)==100:
            figure.update_figure(errors=errors)
            del errors[0]
        


        if glow_bool:
            glow_gradient = new_glow_gradient(glow_gradient=glow_gradient, new_frame=error_frame)
        
        now = time.time()
        color_code = int((round(now-start, 0)%30)/10)
        color_glow = add_glow(frame=set_focus(frame=frame_source, width_corr=width_corr, height_corr=height_corr), 
                              color_code=color_code, 
                              glow_gradient=glow_gradient)
        cv2.imshow("Filter", color_glow)



        if error > 10 or i != 0:
            i += 1
            if i == 1:
                recording = recording_buffer.copy()
            recording.append(frame_source)

            mean_error = np.array(errors[-10:]).mean()

            if i > 100 and mean_error < 3:
                i = 0
                if record_bool:
                    save_video(recording=recording)
                    n_saved += 1
                    print(f"No.{n_saved} video saved!")


   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

        if len(errors)==10:
            glow_bool = not(glow_bool)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            glow_bool = not glow_bool
            print(f"Glow: {glow_bool}")
            glow_gradient = np.zeros(shape=glow_gradient.shape)

        if n_saved == 20:
            break

    # Release the VideoCapture object
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    return n_saved  



if __name__ == "__main__":

    record_bool = True if str(sys.argv[1]) == "-r" else False
    plot_bool = True if str(sys.argv[2]) == "-p" else False
    n_saved = record_video(record_bool=record_bool, plot_bool=plot_bool)

    print(f"Total {n_saved} videos saved!")
