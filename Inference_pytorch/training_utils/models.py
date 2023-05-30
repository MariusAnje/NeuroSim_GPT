from training_utils.qmodules import QNConv2d, QNLinear, NModel
from torch import nn

class QCIFAR(NModel):
    def __init__(self, N=6):
        super().__init__()

        self.conv1 = QNConv2d(N, 3, 64, 3, padding=1)
        self.conv2 = QNConv2d(N, 64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = QNConv2d(N, 64,128,3, padding=1)
        self.conv4 = QNConv2d(N, 128,128,3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = QNConv2d(N, 128,256,3, padding=1)
        self.conv6 = QNConv2d(N, 256,256,3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = QNLinear(N, 256 * 4 * 4, 1024)
        self.fc3 = QNLinear(N, 1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = self.unpack_flattern(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def make_layers(cfg, args):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            conv2d = QNConv2d(args.wl_weight, in_channels, out_channels, kernel_size=v[2], padding=padding)
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d, non_linearity_activation]
            in_channels = out_channels
    return nn.Sequential(*layers)


class VGG(NModel):
    def __init__(self, args, layers):
        super().__init__()
        self.features = layers   
        in_features = layers[-3].op.out_channels * 4 * 4
        
        self.classifier = nn.Sequential(
            QNLinear(args.wl_weight, in_features, 1024),
            nn.ReLU(),
            QNLinear(args.wl_weight, 1024, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.unpack_flattern(x)
        x = self.classifier(x)
        return x



def model_from_cfg_vgg8(cfg, args):
    layers = make_layers(cfg, args)
    model = VGG(args, layers)
    return model