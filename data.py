import torchvision
import os, json

class Pigs:
    def __init__(self, transform):
        self.images = sorted(os.listdir('images2'))
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        image = self.images[i]
        x = torchvision.io.read_image(os.path.join('images2', image))
        d = json.load(open(os.path.join('labels', image[:-3] + 'json')))
        y = d['shapes'][0]['points']
        if d['shapes'][0]['shape_type'] == 'rectangle':
            y = (y[0][0], y[0][1], y[1][0], y[1][1])
        if d['shapes'][0]['shape_type'] == 'polygon':
            y = (min(pt[0] for pt in y), min(pt[1] for pt in y),
                 max(pt[0] for pt in y), max(pt[1] for pt in y))
        y = torchvision.tv_tensors.BoundingBoxes(y, format='XYXY', canvas_size=x.shape[1:])
        if self.transform:
            x, y = self.transform(x, y)
        # normalize 0-1
        y[:, [0, 2]] = y[:, [0, 2]] / y.canvas_size[1]
        y[:, [1, 3]] = y[:, [1, 3]] / y.canvas_size[0]
        # xyxy -> cxcywh
        y = torchvision.ops.box_convert(y, 'xyxy', 'cxcywh')
        return x, y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    ds = Pigs()
    for i, (x, y) in enumerate(ds):
        plt.imshow(x.permute(1, 2, 0))
        plt.gca().add_patch(patches.Rectangle((y[0], y[1]), y[2]-y[0], y[3]-y[1], linewidth=1, edgecolor='r', facecolor='none'))
        plt.title(str(i+1))
        plt.show()
