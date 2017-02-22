from moviepy.editor import *

clip = VideoFileClip("project_video.mp4").cutout(0, 36)
clip.write_videofile("small.mp4")