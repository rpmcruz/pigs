import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

import torch, torchvision
from torchvision.transforms import v2
from time import time
import data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################### DATA ###############################

aug = v2.Compose([
    v2.Resize((512, 512)),
    #v2.RandomCrop((224, 224)),
    #v2.Resize((int(224*1.05), int(224*1.05))),
    #v2.RandomCrop((224, 224)),
    # is hflips and vflips a good idea? - food always in the same place
    #v2.RandomHorizontalFlip(),
    #v2.RandomVerticalFlip(),
    #v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
])
ds = data.Pigs(aug)
#ds = torch.utils.data.Subset(ds, range(10))
ds = torch.utils.data.DataLoader(ds, 8, True, num_workers=2, pin_memory=True)

############################### MODEL ###############################

model = torch.nn.Sequential(
    *list(torchvision.models.resnet50(weights='DEFAULT').children())[:-1],
    torch.nn.Flatten(),
    torch.nn.LazyLinear(4),
    torch.nn.Sigmoid(),
)
model.to(device)

############################### TRAIN ###############################

opt = torch.optim.Adam(model.parameters())
for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    for x, y in ds:
        x = x.to(device)
        y = y.to(device)[:, 0]
        pred = model(x)
        #loss = torch.mean((y-pred)**2)
        #print('y:', y.shape, 'pred:', pred.shape)
        loss = torch.abs(y-pred).mean()
        #loss = torchvision.ops.generalized_box_iou_loss(torchvision.ops.box_convert(y, 'cxcywh', 'xyxy'), torchvision.ops.box_convert(pred, 'cxcywh', 'xyxy')).mean() + \
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss)/len(ds)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Loss: {avg_loss}')

torch.save(model.cpu(), 'model.pth')