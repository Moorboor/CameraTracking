import cv2
import numpy as np
import time
import sys
import seaborn as sns
from tkinter import Tk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os 
import pyautogui



class MotionTracker():
    '''
    Receives the most recent frame.
    Returns the frame difference.

    '''
    def __init__(self, width_corr, height_corr):
        self.frames = []
        self.frame_diffs = []
        self.width_corr, self.height_corr = width_corr, height_corr
        self.kernel = (np.ones((10,10))/100)

    def set_focus(self, *, frame):
        if self.height_corr[1] == 0: 
            self.height_corr[1] = frame.shape[0]
        if self.width_corr[1] == 0: 
            self.width_corr[1] = frame.shape[1]
        return frame[self.height_corr[0]:self.height_corr[1], self.width_corr[0]:self.width_corr[1]]


    def get_frame_diff(self, *, frames):
        mean_frame1 = np.array(frames[:1])
        mean_frame2 = np.array(frames[1:])
        frame_diff = mean_frame1.mean(axis=0) - mean_frame2.mean(axis=0)
        frame_diff = cv2.filter2D(src=frame_diff, 
                                   ddepth=-1, 
                                   kernel=self.kernel) 
        frame_diff = np.where(frame_diff<5, 0, frame_diff)
        diff = (frame_diff**2).mean()

        return frame_diff, diff

    def preprocess_frame(self, *, frame):
        frame = self.set_focus(frame=frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    
    def track_motion(self, *, frame):
        self.frames.append(frame)
        if len(self.frames) > 2: 
            del self.frames[0]

        return self.get_frame_diff(frames=self.frames)  
    
    def render_text(self, *, frame, text="", position=(30,30)):
        text = f"{text:.0f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255) 
        font_thickness = 2
        cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness)
        


class VideoRecorder():

    '''
    
    Receives the most recent frame and keeps it in its buffer.
    Saves videos.

    '''
    def __init__(self):
        self.n_saved = 0
        self.passed_frames = 0
        self.rec_buffer = []
        self.diffs = []
        self.threshold = 3
        self.triggered = False
        self.recording = []

        self.utc_time = datetime.utcnow()
        self.date_string = self.utc_time.strftime('%Y-%m-%d')
        self.time_string = self.utc_time.strftime('%H-%M-%S')
        self.germany_offset = timedelta(hours=1)

        self.year_folder = self.utc_time.strftime('%Y')
        self.month_folder = self.utc_time.strftime('%m')
        self.day_folder = self.utc_time.strftime('%d')
        self.folder_path = os.path.join(self.year_folder, self.month_folder, self.day_folder)

        self.ABS_PATH = os.path.abspath("")
        self.TODAY_RECORDING_PATH = os.path.join(self.folder_path, f'{self.time_string}.mp4')
        os.makedirs(self.folder_path, exist_ok=True)



    def save_video(self):
        utc_time = datetime.utcnow()
        germany_time = utc_time + self.germany_offset
        time_string = germany_time.strftime('%H-%M-%S')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
        out = cv2.VideoWriter(os.path.join(self.folder_path, f"{time_string}.mp4"), fourcc, 20.0, (self.recording[-1].shape[1], self.recording[-1].shape[0]), True)  
        for f in self.recording: 
            out.write(np.array(f, dtype="uint8"))
        out.release()
        self.n_saved += 1
        print(f"No.{self.n_saved} video saved!")
        return True

    def record_video(self,* , frame, diff):
        
        self.rec_buffer.append(frame)
        if len(self.rec_buffer) > 20: 
            del self.rec_buffer[0]
        
        if self.triggered == False:
            self.check_trigger(diff=diff)

        if self.triggered:
            self.recording.append(frame)
            self.passed_frames += 1

        if self.passed_frames > 100:
            self.save_video()
            self.passed_frames = 0
            self.triggered = False

    def check_trigger(self,* , diff):

        if diff > self.threshold or self.passed_frames != 0:
            self.triggered = True
        self.recording = self.rec_buffer.copy()

    def display_n_saved(self):
        print(self.n_saved)



class PartyGlow():

    '''
    Receives the frame difference and turns it into a glow.
    Returns the glow frame.

    '''
    def __init__(self):

        self.color_code = 0
        self.start = time.time()

    def init_frame_glow(self, frame):
        self.frame_glow = np.zeros(shape=(frame.shape[0], frame.shape[1], 3),
                                    dtype="float")


    def get_frame_glow(self, *, frame_diff):

        self.frame_glow *= .1
        self.frame_glow[:, :, self.color_code] = 2 * frame_diff
        self.frame_glow[:,:, 0] += frame_diff
        self.frame_glow[:,:, 1] += frame_diff
        self.frame_glow[:,:, 2] += frame_diff
        return self.frame_glow


    def get_color(self):
        now = time.time()
        self.color_code = int((round(now-self.start, 0)%30)/10)

    def party_glow(self, *, frame_diff):

        self.get_color()
        frame_glow = self.get_frame_glow(frame_diff=frame_diff)
        return frame_glow
    
    def add_party_glow(self, *, frame, party_glow):
        frame = cv2.cvtColor(frame,  cv2.COLOR_GRAY2BGR)
        party_glow = np.array(party_glow, dtype="float")
        frame = np.array(frame, dtype="float")
        return np.array(frame + party_glow, dtype="uint8")



class TimeLapse():

    def __init__(self):
        self.utc_time = datetime.utcnow()
        self.date_string = self.utc_time.strftime('%Y-%m-%d')
        self.time_string = self.utc_time.strftime('%H-%M-%S')
        self.germany_offset = timedelta(hours=2)

        self.year_folder = self.utc_time.strftime('%Y')
        self.month_folder = self.utc_time.strftime('%m')
        self.day_folder = self.utc_time.strftime('%d')
        self.folder_path = os.path.join(self.year_folder, self.month_folder, self.day_folder)

        self.ABS_PATH = os.path.abspath("")
        self.TODAY_RECORDING_PATH = os.path.join(self.folder_path, f'{self.time_string}.mp4')
        os.makedirs(self.folder_path, exist_ok=True)

        self.n_saved = 0
        self.counter = 0
        self.interval = 1000

    def check(self, frame):
        self.counter += 1
        if self.counter % self.interval == 0:
            self.save_photo(frame) 

    def save_photo(self, frame):

        utc_time = datetime.utcnow()
        germany_time = utc_time + self.germany_offset
        time_string = germany_time.strftime('%H-%M-%S')
        
        cv2.imwrite(os.path.join(self.folder_path, f"{time_string}.jpg"), frame)
        self.n_saved += 1
        print(f"No.{self.n_saved} photo saved!")

    

class Figure():

    '''
    
    Receives the frame difference and turns it into statistics.
    Displays the fiure on top of the shown frame.

    '''
    def __init__(self, overlay=False):

        self.timesteps = 25
        self.n_values = 100
        self.diffs = []
        self.central_moments = [[] for _ in range(4)]
        
        if overlay:
            root = Tk()
            root.withdraw()  # Hide the dummy root window
            plt.rcParams['toolbar'] = 'None'    
            plt.get_current_fig_manager().window.attributes('-alpha', 0.3)             # Remove bars
            plt.get_current_fig_manager().window.overrideredirect(True)
            # plt.get_current_fig_manager().window.title("")
            
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(8,8))
        colors = sns.color_palette("rocket", 3)[1:]
        self.fig.patch.set_facecolor('#000000')
        self.ax1.set_facecolor('#000000')
        self.ax2.set_facecolor('#000000')


        (self.ln1,) = self.ax1.plot(range(self.n_values), np.linspace(0,150, self.n_values), animated=True, c=colors[0])
        (self.ln2,) = self.ax2.plot(np.linspace(-50, 350, self.n_values), np.linspace(0,0.35, self.n_values), animated=True, c=colors[1])
        # self.ax2.fill_between(np.linspace(-50, 350, self.n_values), 0, 0, color='blue', label="filling")

        plt.show(block=False)
        plt.pause(0.1)
        # plt.get_current_fig_manager().window.state('zoomed')
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.ax1.draw_artist(self.ln1)
        self.ax2.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)


    def get_current_values(self, *, diff):
        def get_central_moments(*, x, k):
            return ((x-x.mean())**k).mean()
        
        def get_distribution(*, x, mu, std):
            return (1/std*(2*np.pi)**0.5) * np.exp(-0.5*((x-mu)/std)**2)
        
        if len(self.diffs) == self.n_values:
            del self.diffs[0]
        self.diffs.append(diff)
        diffs = np.array(self.diffs)

        mean = np.array(diffs).mean()
        std = get_central_moments(x=diffs, k=2)**0.5
        skewness = get_central_moments(x=diffs, k=3)/(get_central_moments(x=diffs, k=2)**(0.5*3))
        kurtosis = get_central_moments(x=diffs, k=4)/(get_central_moments(x=diffs, k=2)**(0.5*(4)))
        dist = get_distribution(x=np.linspace(-50,350, self.n_values),mu=mean, std=std)

        self.central_moments[0].append(mean)
        self.central_moments[1].append(std)
        self.central_moments[2].append(skewness)
        self.central_moments[3].append(kurtosis)

        for central_moment in self.central_moments:
            if len(central_moment)==25:
                del central_moment[0]

        return dist

    def update_figure(self, *, diff):

        dist = self.get_current_values(diff=diff)
        self.fig.canvas.restore_region(self.bg)

        self.ln1.set_ydata(self.diffs)
        self.ln2.set_ydata(dist)

        # re-render the artist, updating the canvas state, but not the screen
        # path = self.ax2.fill_between(np.linspace(-50, 350, self.n_values), 0, dist, label="filling", alpha=0.8)

        if len(self.diffs) == self.n_values:
            # self.ax2.draw_artist(path)
            self.ax1.draw_artist(self.ln1)
            self.ax2.draw_artist(self.ln2)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()



class BackgroundVideo():

    def __init__(self):
        self.ABS_PATH = os.path.abspath("")
        self.video_path = os.path.join(self.ABS_PATH, "BackgroundVideo", "Render4.mp4")
        self.video = self.load_video()
        self.frame_idx = 0

    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        ret = True
        while ret:
            ret, img = cap.read()
            if ret:
                frames.append(img)
        self.video = np.stack(frames, axis=0, dtype="uint8") 
        cap.release()

    def current_frame(self):
        if self.frame_idx == 499:
            self.frame_idx = 0
        self.frame_idx += 1
        return self.video[self.frame_idx]
    


class Camera():

    def __init__(self, cam):

        self.cam = cam

        if cam:
            # self.cap = cv2.VideoCapture(2) # Laptop 
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
            # self.cap.set(cv2.CAP_PROP_EXPOSURE, 10) 
            # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # auto mode
        
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        
            # cv2.namedWindow("Filter", cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty("Filter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def read(self):
        ret, src_frame = self.cap.read() if self.cam else (True, np.array(pyautogui.screenshot()))
        if not ret:
            print("Error: cap not found")
            self.quit()
        return src_frame
        
    
    def quit(self):
        if self.cam:
            self.cap.release()
            cv2.destroyAllWindows()
        quit()



class Trajectory():

    def __init__(self, n_points=10):
        self.n_points = n_points
        self.trajectory = []

    def get_trajectory(self, *, frame):

        
        if len(self.trajectory) > self.n_points:
            del self.trajectory[0]



class MotionTrackerManager():

    def __init__(self, *, mtracker=True, rec=False, party=False, fig=False, bgv=False, t_lapse=False, cam=True):

        self.mtracker, self.rec, self.party, self.fig, self.bgv, self.t_lapse, self.cam = mtracker, rec, party, fig, bgv, t_lapse, cam
        self.width_corr, self.height_corr = [0, 500], [0, 500] # [150, 250] # width, height

        self.camera = Camera(cam=self.cam)    
        if mtracker: self.motionTracker = MotionTracker(self.width_corr, self.height_corr)
        if rec: self.videoRecorder = VideoRecorder()
        if party: self.partyGlow = PartyGlow()
        if fig: self.figure = Figure() 
        if bgv: self.backgroundVideo = BackgroundVideo() # self.b_video = self.backgroundVideo.load_video()
        if t_lapse: self.timelapse = TimeLapse()

    
    
    def run(self):
    
        while True:

            src_frame = self.camera.read()

            if self.mtracker:
                frame = self.motionTracker.preprocess_frame(frame=src_frame)
                frame_diff, diff = self.motionTracker.track_motion(frame=frame)
            if self.t_lapse:
                self.timelapse.check(frame=src_frame)
            if self.rec:
                self.videoRecorder.record_video(frame=src_frame, diff=diff)
            if self.fig:
                self.figure.update_figure(diff=diff)
            if self.party:
                party_glow = self.partyGlow.party_glow(frame_diff=frame_diff)
                frame = self.partyGlow.add_party_glow(frame=frame, party_glow=party_glow)

                if self.bgv:
                    frame = self.backgroundVideo.current_frame()
                    party_glow = np.array(party_glow, dtype="float")
                    frame = np.array(frame, dtype="float")
                    frame += party_glow*3
                    frame = np.where(frame>255, 255, frame)
                    frame = np.array(frame, dtype="uint8")

            debug_frame = src_frame.copy()
            cv2.rectangle(debug_frame, (self.width_corr[0], self.height_corr[0]), (self.width_corr[1], self.height_corr[1]), (0,255,0))
            cv2.imshow("Source Frame", debug_frame)
            cv2.imshow("Filter", frame_diff)

    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.camera.quit() 
            if cv2.waitKey(1) & 0xFF == ord("1"):
                self.partyGlow = PartyGlow()
                self.partyGlow.init_frame_glow(frame=frame)
                self.party = not(self.party)
                
            

if __name__ == "__main__":
    rec = True if str(sys.argv[1]) == "-r" else False
    party = False
    fig = False
    bgv = False
    t_lapse = False
    motionTrackerManager = MotionTrackerManager(
                                                rec=rec,
                                                party=party,
                                                fig=fig, 
                                                bgv=bgv,
                                                t_lapse=t_lapse)
    motionTrackerManager.run()
