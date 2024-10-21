# https://www.sciencedirect.com/science/article/abs/pii/S0168169919319556

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--root', default='/data/bioeng')
args = parser.parse_args()

from torchvision.transforms import v2
import torch
import matplotlib.pyplot as plt
import data, xai
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## DATA ##############################

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 30 fps is reduced to 15 fps (by skip_frames=2)
# 2 seconds sequences are then used
ts = data.Video(data.Pigs(args.root), 2*15, 2, transforms)
ts = torch.utils.data.DataLoader(ts)

############################## MODEL ##############################

model = torch.load(args.model, map_location=device)

############################## LOOP ##############################

for frames, is_violent in ts:
    frames_cuda = frames.to(device)
    preds = model(frames_cuda)
    map = xai.GradCAM(model, model.encoder.features[28], None, frames, is_violent)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(frames[0, -1].permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(map[0].cpu())
    plt.suptitle(f'Label: {is_violent[0]} Pred: {preds[0].cpu()}')
    plt.draw()
    plt.pause(0.0001)