import subprocess
import os

ABS_PATH = os.path.abspath("")



def render_blender():
    blender_executable = r"C:\Program Files\Blender Foundation\Blender 4.0\blender-launcher.exe"
    blender_script = os.path.join(ABS_PATH, f"blender_script.py")

    # command = [blender_executable]
    command = [blender_executable, "--background", "--python", blender_script]
    subprocess.run(command)

    return True


render_blender()