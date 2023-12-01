import bpy
import os
from datetime import datetime, timedelta

ABS_PATH = os.path.abspath("")
RENDER_PATH = os.path.join(ABS_PATH, "Renders")
os.makedirs(RENDER_PATH, exist_ok=True)

utc_time = datetime.utcnow()
date_string = utc_time.strftime('%Y-%m-%d')



def render_video(output_path):
    # Set rendering settings
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    
    # Set output path and filename
    bpy.context.scene.render.filepath = output_path

    # Set frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 250  # Set your desired end frame

    # Render animation
    bpy.ops.render.render(animation=True)
    return True



# Replace 'output_video.mp4' with your desired output file path and name
render_video(os.path.join(RENDER_PATH, f"{date_string}.mp4"))
