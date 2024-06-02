import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

import torch
from torchvision.transforms import v2
import data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################### DATA ###############################

aug = v2.Compose([
    v2.Resize((512, 512)),
    v2.ToDtype(torch.float32, True),
])
ds = data.Pigs(aug)
ds = torch.utils.data.DataLoader(ds, 1, num_workers=2, pin_memory=True)

############################### MODEL ###############################

model = torch.load('model.pth', map_location=device)

############################### EVAL ###############################

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import numpy as np
import imageio
frames = []

for i, (x, y) in enumerate(tqdm(ds)):
    x = x.to(device)
    with torch.no_grad():
        ŷ = model(x)
    ŷ = ŷ[0].cpu() * torch.tensor(x.shape[2:]).repeat(2)
    y = y[0, 0] * torch.tensor(x.shape[2:]).repeat(2)
    plt.clf()
    plt.imshow(x[0].cpu().permute(1, 2, 0))
    plt.gca().add_patch(patches.Rectangle((y[0]-y[2]/2, y[1]-y[3]/2), y[2], y[3], linewidth=2, edgecolor='r', facecolor='none'))
    plt.gca().add_patch(patches.Rectangle((ŷ[0]-ŷ[2]/2, ŷ[1]-ŷ[3]/2), ŷ[2], ŷ[3], linewidth=2, edgecolor='g', facecolor='none'))
    plt.title(str(i+1))
    plt.draw()
    frame = np.frombuffer(plt.gca().figure.canvas.tostring_rgb(), dtype='uint8')
    frame = frame.reshape(plt.gca().figure.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)

imageio.mimsave('animation.gif', frames, fps=3, repeat=True)
