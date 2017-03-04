t1 = 10
t2 = 30

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("project_video.mp4", t1, t2, targetname="test.mp4")