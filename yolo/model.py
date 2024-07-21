import torch
import torch.nn as nn

dropout = 0.2

architecture_config = [
(7, 64, 2,3),
"M",
(3, 192, 1, 1),
"M",
(1, 128, 1, 0),
(3, 256, 1, 1),
(1, 256, 1, 0),
(3, 512, 1, 1),
"M",
[(1, 256, 1, 0), (3, 512, 1, 1), 4],
(1, 512, 1, 0),
(3, 1024, 1, 1),
"M",
[(1, 512, 1, 0), (3, 1024, 1, 1), 2],
(3, 1024, 1, 1),
(3, 1024, 2, 1),
(3, 1024, 1, 1),
(3, 1024, 1, 1),
]



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.architecture = architecture_config
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        print(f'shape of x after convolution layers is {x.shape}')
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        in_channels = self.in_channels
        layers = []
        for layer in architecture:
            if type(layer) == tuple:
                layers += [CNNBlock(in_channels, layer[1], kernel_size = layer[0], stride = layer[2], padding = layer[3])]
                in_channels = layer[1]
            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))]
            elif type(layer) == list:
                conv1 = layer[0]
                conv2 = layer[1]
                repeat = layer[2]

                for count in range(repeat):
                    layers += [CNNBlock(in_channels, conv1[1], kernel_size = conv1[0], stride = conv1[2], padding = conv1[3])]

                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size = conv2[0], stride = conv2[2], padding = conv2[3])]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)


    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 *S * S, 512), #original 4096
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(512, S*S*(C + B*5)),
        )


def test(split_size=7, num_boxes = 2, num_classes=20):
    model = Yolov1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

test()

        










    







