import torch, torchvision

class Chen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.vgg16(weights='DEFAULT')
        self.lstm = torch.nn.LSTM(1000, 256, batch_first=True)
        self.clf = torch.nn.Linear(256, 1)

    def forward(self, x):
        shape = x.shape
        x = self.encoder(torch.flatten(x, 0, 1))
        x = x.reshape(shape[0], shape[1], -1)
        x = self.lstm(x)[0][:, -1]
        return torch.sigmoid(self.clf(x))