import subprocess


def render_blender():
    blender_executable = r"C:\Program Files\Blender Foundation\Blender 4.0\blender-launcher.exe"
    blender_script = r"C:\Users\moorb\OneDrive\Documents\MyProjects\Code\Repositories\CameraTracking\blender_script.py"

    # command = [blender_executable]
    command = [blender_executable, "--background", "--python", blender_script]
    subprocess.run(command)

    return True


render_blender()