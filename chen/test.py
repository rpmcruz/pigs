# https://www.sciencedirect.com/science/article/abs/pii/S0168169919319556

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--root', default='/data/bioeng')
args = parser.parse_args()

from torchvision.transforms import v2
import torch, torchmetrics
import data
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
ts = torch.utils.data.DataLoader(ts, args.batchsize, num_workers=4, pin_memory=True)

############################## MODEL ##############################

model = torch.load(args.model, map_location=device)

############################## LOOP ##############################

acc = torchmetrics.classification.BinaryAccuracy().to(device)
for frames, is_violent in ts:
    frames = frames.to(device)
    is_violent = is_violent.to(device)
    preds = model(frames)[:, 0]
    acc.update(preds, is_violent)
print('Accuracy:', acc.compute())