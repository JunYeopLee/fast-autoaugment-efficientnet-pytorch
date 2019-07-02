import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self, backbone, args):
        super(BaseNet, self).__init__()

        # Separate layers
        self.first = nn.Sequential(*list(backbone.children())[:1])
        self.after = nn.Sequential(*list(backbone.children())[1:-1])
        self.fc = list(backbone.children())[-1]

        self.img_size = (224, 224)

    def forward(self, x):
        f = self.first(x)
        x = self.after(f)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x, f
