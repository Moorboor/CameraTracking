import bpy

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

# Replace 'output_video.mp4' with your desired output file path and name
render_video(r"path/to/your/output/output_video.mp4")
