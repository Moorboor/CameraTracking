import bpy
import os
from datetime import datetime

ABS_PATH = os.path.abspath("")
RENDER_PATH = os.path.join(ABS_PATH, "Renders")
INPUT_PATH = os.path.join(ABS_PATH, "2023", "12", "01")
os.makedirs(RENDER_PATH, exist_ok=True)



def move_objects():
    for obj in bpy.data.objects: print(obj.name)


def render_video(output_path):
    # Set rendering settings
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    
    # Set output path and filename
    bpy.context.scene.render.filepath = output_path

    # Set frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 10  # Set your desired end frame

    # Render animation
    bpy.ops.render.render(animation=True)
    return True

video_names = os.listdir(INPUT_PATH)
video_path = r"C:\Users\moorb\OneDrive\Desktop\PXL_20240327_121126161.mp4"


# render_video(os.path.join(RENDER_PATH, f"test.mp4"))
move_objects()