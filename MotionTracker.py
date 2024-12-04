import cv2
import numpy as np
import time
from datetime import datetime
import os 
import seaborn as sns
from tkinter import Tk
import matplotlib.pyplot as plt
import subprocess
# import pyautogui

# from gpiozero import AngularServo
# import RPi.GPIO as GPIO
# from RPLCD import CharLCD
# from picamera2 import Picamera2


class MotionTracker():

    def __init__(self, width_corr, height_corr):
        self.width_corr, self.height_corr = width_corr, height_corr
        self.frames = []
        self.frame_diffs = []
        self.kernel = (np.ones((10,10))/100)

    def set_focus(self, *, frame):
        if self.height_corr[1] == 0: 
            self.height_corr[1] = frame.shape[0]
        if self.width_corr[1] == 0: 
            self.width_corr[1] = frame.shape[1]
        return frame[self.height_corr[0]:self.height_corr[1], self.width_corr[0]:self.width_corr[1]]

    def get_frame_diff(self, *, frames):
        frame_diff = frames[0] - frames[-1] 
        frame_diff = np.clip(frame_diff, 0, 255)
        _, frame_diff = cv2.threshold(frame_diff, 5, 255, cv2.THRESH_TOZERO)
        frame_diff = cv2.erode(frame_diff, kernel=self.kernel)
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

    def __init__(self):
        self.n_saved = 0
        self.threshold = 3
        self.passed_frames = 0
        self.rec_buffer = []
        self.recording = []
        self.diffs = []
        self.triggered = False
        self.ABS_PATH = os.path.abspath("")


    def save_video(self):
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        TODAY_PATH = os.path.join(self.ABS_PATH, "Recordings", "Videos", f"{date}")
        os.makedirs(TODAY_PATH, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(TODAY_PATH, f"{now.strftime('%H-%M-%S')}.mp4"), fourcc, 20.0, (self.recording[-1].shape[1], self.recording[-1].shape[0]), isColor=True)  
        
        for f in self.recording: 
            out.write(np.array(f, dtype="uint8"))
        out.release()

        self.n_saved += 1
        print(f"Video: No.{self.n_saved} saved.")
        

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



class PartyGlow():

    def __init__(self):
        self.color_code = 0
        self.start = time.time()

    def init_frame_glow(self, frame):
        self.frame_glow = np.zeros(shape=(frame.shape[0], frame.shape[1], 3), dtype="float")

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


class Timelapse():

    def __init__(self, pi):
        self.pi = pi
        self.n_saved = 0
        self.counter = 0
        self.interval = 10.0
        self.start = time.time()
        self.ABS_PATH = os.path.abspath("")

    def save_photo(self, frame):

        self.diff = time.time() - self.start
        if self.diff > self.interval:
            now = datetime.now()  
            date = now.strftime('%Y-%m-%d')
            TODAY_PATH = os.path.join(self.ABS_PATH, "Recordings", "Timelapse", f"{date}")
            os.makedirs(TODAY_PATH, exist_ok=True)
            if self.pi:
                subprocess.run("rpicam-still", "--raw", "--output", f"{now.strftime('%H-%M-%S')}.jpg", shell=True, check=True)
            else:
                cv2.imwrite(os.path.join(TODAY_PATH, f"{now.strftime('%H-%M-%S')}.jpg"), frame)
            self.n_saved += 1
            print(f"Timelapse: No.{self.n_saved} saved.")
            self.start = time.time()


class Figure():

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
            
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(16,16))
        # plt.get_current_fig_manager().window.state('zoomed')
        colors = sns.color_palette("rocket", 3)[1:]
        # self.fig.patch.set_facecolor('#000000')
        # self.ax1.set_facecolor('#000000')
        # self.ax2.set_facecolor('#000000')
        (self.ln1,) = self.ax1.plot(range(self.n_values), np.linspace(0,1, self.n_values), animated=True, c=colors[0])
        (self.ln2,) = self.ax2.plot(np.linspace(-0.5, 2, self.n_values), np.linspace(0,0.35, self.n_values), animated=True, c=colors[1])
        # self.ax2.fill_between(np.linspace(-50, 350, self.n_values), 0, 0, color='blue', label="filling")

        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax1.draw_artist(self.ln1)
        self.ax2.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)

    def get_current_dist(self, *, x):
        def get_central_moments(*, x, k):
            return ((x-x.mean())**k).mean()
        
        def get_distribution(*, x, mu, std):
            return (1/std*(2*np.pi)**0.5) * np.exp(-0.5*((x-mu)/std)**2)
        
        if len(self.diffs) == self.n_values:
            del self.diffs[0]
        self.diffs.append(x)
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

    def update_figure(self, *, x):

        self.fig.canvas.restore_region(self.bg)
        dist = self.get_current_dist(x=x)
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



class ServoMotor():

    def __init__(self, resolution):
        self.resolution = [r/2 for r in resolution]
        self.boundaries = 90
        self.threshold = 100
        
        self.curr_angle = 0
        self.servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

    def move(self, coor):
        diff = [r-c for c, r in zip(coor, self.resolution)]

        if (diff[0]>self.threshold) and ((self.curr_angle-2)>-self.boundaries):
            self.curr_angle -= 2
        elif (diff[0]<-self.threshold) and ((self.curr_angle+2)<self.boundaries):
            self.curr_angle += 2

        if abs(self.servo.angle-self.curr_angle) > 20:
            self.servo.angle = self.curr_angle
            

class LCD():

    def __init__(self):
        self.lcd = CharLCD(cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[33, 31, 29, 23], numbering_mode=GPIO.BOARD)
        self.lcd.clear()
    
    def update(self, text):
        self.lcd.clear()
        self.lcd.write_string(f"Curr angle: {text:.0f}")


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

    def __init__(self, cam, pi):
        self.cam, self.pi = cam, pi
        self.width, self.height = 1280, 720

        if self.cam:
            if self.pi:
                self.cap = Picamera2()
                video_config = self.cap.create_video_configuration(main={"size": (self.width, self.height), "format": "BGR888"})
                self.cap.configure(video_config)
                self.cap.start()
            else:
                # self.cap = cv2.VideoCapture(0) # Laptop 
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
                # self.cap.set(cv2.CAP_PROP_EXPOSURE, 10) 
                # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # auto mode
            
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            
                # cv2.namedWindow("Filter", cv2.WINDOW_NORMAL)
                # cv2.setWindowProperty("Filter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    def read(self):
        if self.cam:
            ret, src_frame = (True, self.cap.capture_array()) if self.pi else self.cap.read()
        else:
            ret, src_frame = (True, pyautogui.screenshot())
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
        self.trajectory_list = []
    
    def get_image_moments(self, *, frame):
        _, frame_mask = cv2.threshold(frame, 70, 255, 0)
        moments = cv2.moments(frame_mask)
        x_mean = moments["m10"]/(moments["m00"]+1e-06)
        y_mean = moments["m01"]/(moments["m00"]+1e-06)

        self.trajectory_list.append([int(x_mean), int(y_mean)])

        if len(self.trajectory_list) > self.n_points:
            del self.trajectory_list[0]
    

class Lambda():

    def __init__(self, lamb_func=lambda x: (x)):
        self.lamb_func = lamb_func

    def __call__(self, lamb_func, frame):
        return lamb_func(frame)


class MotionTrackerManager():

    def __init__(self, *, mtracker=True, rec=False, party=False, fig=False, bgv=False, t_lapse=False, cam=True, trajec=False, pi=False, servo=False, lcd_bool=False, lamb_bool=False):
        self.mtracker = mtracker
        self.rec = rec
        self.party = party
        self.fig = fig
        self.bgv = bgv
        self.t_lapse = t_lapse
        self.cam = cam
        self.trajec = trajec
        self.pi = pi
        self.servoMotor_bool = servo
        self.lcd_bool = lcd_bool
        self.lamb_bool = lamb_bool

        self.width_corr, self.height_corr = [0, 0], [0, 0] # [150, 250] # width, height

        self.camera = Camera(cam=self.cam, pi=self.pi)    
        if self.mtracker: self.motionTracker = MotionTracker(self.width_corr, self.height_corr)
        if self.rec: self.videoRecorder = VideoRecorder()
        if self.party: self.partyGlow = PartyGlow()
        if self.fig: self.figure = Figure() 
        if self.bgv: self.backgroundVideo = BackgroundVideo() # self.b_video = self.backgroundVideo.load_video()
        if self.t_lapse: self.timelapse = Timelapse(pi=self.pi)
        if self.trajec: self.trajectory = Trajectory()
        if self.servoMotor_bool: self.servoMotor = ServoMotor((self.camera.width, self.camera.height))
        if self.lcd_bool: self.lcd = LCD()
        if self.lamb_bool: self.lamb = Lambda()

    
    
    def run(self):
    
        while True:
            src_frame = self.camera.read()

            if self.lamb_bool: 
                frame_test = self.lamb(self.lamb_func, frame=src_frame.copy())
            if self.mtracker:
                frame = self.motionTracker.preprocess_frame(frame=src_frame.copy())
                frame_diff, diff = self.motionTracker.track_motion(frame=frame)
            if self.t_lapse:
                self.timelapse.save_photo(frame=src_frame)
            if self.rec:
                self.videoRecorder.record_video(frame=src_frame, diff=diff)

            if self.trajec:
                self.trajectory.get_image_moments(frame=frame_diff)
                if self.servoMotor_bool:
                    self.servoMotor.move(coor=(self.trajectory.trajectory_list[-1]))
                if self.lcd_bool:
                    self.lcd.update(text=self.servoMotor.curr_angle)

            if self.fig:
                self.figure.update_figure(x=diff)
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

            # cv2.rectangle(debug_frame, (self.width_corr[0], self.height_corr[0]), (self.width_corr[1], self.height_corr[1]), (0,255,0))
            if self.trajec:
                cv2.rectangle(src_frame, (int(self.trajectory.trajectory_list[-1][0]), int(self.trajectory.trajectory_list[-1][1])), (int(self.trajectory.trajectory_list[-1][0])+10, int(self.trajectory.trajectory_list[-1][1])+10), (0,255,0), thickness=10)
            if self.lamb_bool:
                cv2.imshow("Lambda", frame_test)
            # cv2.imshow("Filter", frame_diff)
            cv2.imshow("Source", src_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.camera.quit() 
            if cv2.waitKey(1) & 0xFF == ord("1"):
                self.partyGlow = PartyGlow()
                self.partyGlow.init_frame_glow(frame=frame)
                self.party = not(self.party)
                

if __name__ == "__main__":
    motionTrackerManager = MotionTrackerManager(
        mtracker=True,
        rec=False,
        party=False,
        fig=False,
        bgv=False,
        t_lapse=False,
        cam=True,
        trajec=True,
        pi=False,
        servo=False,
        lcd_bool=False,
        lamb_bool=False)
    motionTrackerManager.run()
