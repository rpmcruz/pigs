import imageio
import pandas as pd
import torch
from bisect import bisect
from itertools import accumulate
import os

violent_behaviors = ('BITE', 'HEAD BODY KNOCK', 'NOSING', 'PRESSING PARALLEL / INVERSE', 'BLOCK OF FEEDING')

class Pigs:
    def __init__(self, root):
        path = os.path.join(root, 'pigs', 'videosPenA_2018_12_04_9h04_12h14')
        self.videos = [imageio.get_reader(os.path.join(path, f), 'ffmpeg') for f in sorted(os.listdir(path)) if f.endswith('.mp4')]
        self.video_fps = [v.get_meta_data()['fps'] for v in self.videos]
        self.video_cum_durations = list(accumulate((v.get_meta_data()['duration'] for v in self.videos), initial=0))
        self.video_cum_frames = list(accumulate((v.get_meta_data()['duration']*v.get_meta_data()['fps'] for v in self.videos), initial=0))
        labels = pd.read_excel(os.path.join(path, 'labels', 'Observação Pen A.xlsx'), 'Total', usecols=['Time', 'Subject', 'Behavior', 'Status'])
        self.ann_times = [0]
        self.ann_behaviors = [['IDLE']*10]
        for _, (time, subject, behavior, status) in labels.iterrows():
            if type(subject) != str: continue
            pig = int(subject.split()[0][1:])-1
            behaviors = self.ann_behaviors[-1]
            if status in ('START', 'STOP'):
                behavior = behavior if status == 'START' else 'IDLE'
                self.ann_times += [time]
                self.ann_behaviors += [behaviors[:pig] + [behavior] + behaviors[pig+1:]]
            elif status == 'POINT':
                self.ann_times += [time-2.5, time+2.5]
                self.ann_behaviors += [behaviors[:pig] + [behavior] + behaviors[pig+1:],
                    behaviors[:pig] + ['IDLE'] + behaviors[pig+1:]]

    def __len__(self):
        return int(self.video_cum_frames[-1])

    def __getitem__(self, index):
        # index -> video, frame
        video_index = bisect(self.video_cum_frames, index)-1
        video_frame = index - int(round(self.video_cum_frames[video_index]))
        frame = torch.tensor(self.videos[video_index].get_data(video_frame)).permute(2, 0, 1)
        # index -> time
        video_index = bisect(self.video_cum_frames, index)-1
        time = self.video_cum_durations[video_index] + (index-self.video_cum_frames[video_index])/self.video_fps[video_index]
        ann_index = bisect(self.ann_times, time)-1
        behaviors = self.ann_behaviors[ann_index]
        violent = any(b in violent_behaviors for b in behaviors)
        return frame, time, behaviors, violent

class Video:
    def __init__(self, ds, sequence_len, skip_frames=1, transforms=None):
        self.ds = ds
        self.sequence_len = sequence_len
        self.skip_frames = skip_frames
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)//self.skip_frames - self.sequence_len

    def __getitem__(self, i):
        seq = [self.ds[(i+s)*self.skip_frames] for s in range(self.sequence_len)]
        video = torch.stack([s[0] for s in seq])
        if self.transforms:
            video = self.transforms(video)
        return video, seq[-1][-1]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    pigs = Pigs('/data/bioeng')
    print('len pigs:', len(pigs))
    subjects = ['Grey Black spread Dorso - Full Back', 'White', 'Black Mancha RT Thigh and Flank', 'White', 'White', 'White', 'White', 'Black spots With Light Grey Background', 'Black Mancha Neck and Thorax With Spots Lumber', 'Black Large Spots With White Background']
    for i in range(850*30, len(pigs), 30):
        frame, time, behaviors, violent = pigs[i]
        plt.clf()
        plt.imshow(frame.permute(1, 2, 0))
        plt.title(f'frame {i} time {time:.0f}')
        for pig, behavior in enumerate(behaviors):
            c = 'white' if behavior == 'LYING' else 'red'
            plt.text(0, 100+pig*30, f'{behavior} {pig+1} {subjects[pig]}', c=c)
        plt.draw()
        plt.pause(0.0001)