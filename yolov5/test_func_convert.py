import moviepy.editor as moviepy
clip = moviepy.VideoFileClip("continuous_video.avi")
clip.write_videofile("continuous_video.mp4")