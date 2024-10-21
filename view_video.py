import pandas as pd
import imageio
import os
import matplotlib.pyplot as plt

path = '/data/bioeng/pigs/videosPenA_2018_12_04_9h04_12h14/labels/Observação Pen A.xlsx'
df = pd.read_excel(path, 'Total', index_col='Time', usecols=['Time', 'Subject', 'Behavior', 'Status'])
df = df.dropna(subset=['Subject'])
df['Subject'] = [int(s.split()[0][1:])-1 for s in df['Subject']]

subjects = ['Grey Black spread Dorso - Full Back', 'White', 'Black Mancha RT Thigh and Flank', 'White', 'White', 'White', 'White', 'Black spots With Light Grey Background', 'Black Mancha Neck and Thorax With Spots Lumber', 'Black Large Spots With White Background']
pigs = ['LYING']*10
path = '/data/bioeng/pigs/videosPenA_2018_12_04_9h04_12h14'

videos = [imageio.get_reader(os.path.join(path, f), 'ffmpeg') for f in sorted(os.listdir(path)) if f.endswith('.mp4')]
videos_times = [video.get_meta_data()['duration'] for video in videos]

def on_press(event):
    if event.key == 'enter':
        global next_video
        next_video = True
next_video = False
current_time = 0
for fname in videos:
    video = imageio.get_reader(os.path.join(path, fname), 'ffmpeg')
    fps = video.get_meta_data()['fps']
    for i, frame in enumerate(video):
        current_time += 1/fps
        plt.imshow(frame)
        time = i/30
        plt.text()
        plt.title(fname)
        plt.draw()
        plt.pause(0.0001)
        plt.gcf().canvas.mpl_connect('key_press_event', on_press)
        if next_video:
            next_video = False
            break
    cum_i += video.count_frames()