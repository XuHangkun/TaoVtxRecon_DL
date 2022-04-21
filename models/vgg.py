import torch
import torch.nn as nn

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):

    def __init__(self,
            inchannel,
            out_num,
            dropout = 0.0,
            init_weights = True):
        super().__init__()
        self.inchannel = inchannel
        self.out_num = out_num
        self.dropout = dropout
        self.feature = make_layers(cfg["E"], inchannel)
        self.recon = nn.Sequential(
            nn.Linear(512 * 2 * 4, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, out_num),
        )

        if init_weights:
            self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def save(self, path):
        kwargs = {"inchannel" : self.inchannel, "out_num" : self.out_num}
        ckpt = {"kwargs" : kwargs, "state_dict" : self.state_dict()}
        torch.save(ckpt, path)

    def load(self, ckpt):
        self.load_state_dict(ckpt["state_dict"])

    def forward(self, x):
        """
        Args:
            x : [N, Cin, H, W]
        Return:
            y : [N, Cout]
        Return:

        """
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.recon(x)
        return x * 900

def test():
    model = VGG(1,3)
    import torch
    img = torch.rand([256,1,128,64])
    out = model(img)
    print(f"Input : {img.shape}\nOutput : {out.shape}")

if __name__ == "__main__":
    test()
