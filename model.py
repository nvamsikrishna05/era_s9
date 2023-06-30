import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        dropout_value = 0.1

        # Convolution Block 1
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # RF: 3 n_out: 

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 5

        # Transition Block 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
        ) # RF: 5

        # self.c4 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(32),
        #     nn.Dropout(dropout_value)
        # ) # RF: 7

        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # RF: 7

        # Convolution Block 2
        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 11

        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 15

        # Transition Block 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
        ) # RF: 15

        self.c8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # RF: 19

        # Convolution Block 3
        self.c9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 27

        self.c10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # RF: 35


        # Transition Block 3
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, padding=0),
        ) # RF: 35

        self.c12 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # RF: 43

        # Convolution Block 4
        self.c13 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 59

        self.c14 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # RF: 75

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=10, bias=False)
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = self.c11(x)
        x = self.c12(x)
        x = self.c13(x)
        x = self.c14(x)
        # print(f"Shape = \n {x.shape}")
        x = self.gap(x)

        x = x.view(-1, 128)
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)
    

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        dropout_value = 0.1

        # Convolution Block 1
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # RF: 3 n_out: 

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 5

        # Transition Block 1
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, padding=0),
        ) # RF: 5

        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) # RF: 7

        # Convolution Block 2
        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # RF: 11

        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 15

        # Transition Block 2
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
        ) # RF: 15

        self.c8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # RF: 19

        # Convolution Block 3
        self.c9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 27

        self.c10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        ) # RF: 35


        # Transition Block 3
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, padding=0),
        ) # RF: 35

        self.c12 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # RF: 43

        # Convolution Block 4
        self.c13 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # RF: 59

        self.c14 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=20)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=10, bias=False)
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = self.c11(x)
        x = self.c12(x)
        x = self.c13(x)
        x = self.c14(x)
        x = self.gap(x)

        x = x.view(-1, 256)
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)