# https://www.sciencedirect.com/science/article/abs/pii/S0168169919319556

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--root', default='/data/bioeng')
args = parser.parse_args()

from torchvision.transforms import v2
import torch
from tqdm import tqdm
from time import time
import data, models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## DATA ##############################

transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 30 fps is reduced to 15 fps (by skip_frames=2)
# 2 seconds sequences are then used
tr = data.Video(data.Pigs(args.root), 2*15, 2, transforms)
tr = torch.utils.data.DataLoader(tr, args.batchsize, True, num_workers=4, pin_memory=True)

############################## MODEL ##############################

model = models.Chen()
model.to(device)

############################## LOOP ##############################

opt = torch.optim.AdamW(model.parameters())
for epoch in tqdm(range(args.epochs)):
    tic = time()
    avg_loss = 0
    for frames, is_violent in tr:
        frames = frames.to(device)
        is_violent = is_violent.to(device)
        preds = model(frames)[:, 0]
        loss = torch.nn.functional.binary_cross_entropy(preds, is_violent.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss)/len(tr)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f} - Avg loss: {avg_loss}')

torch.save(model.cpu(), 'model.pth')