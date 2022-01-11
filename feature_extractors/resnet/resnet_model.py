import torch.nn as nn

from .resblock import ResBlock


class ResNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,
                               padding=self.calc_pad(kernel_size=3))

        self.dropout = nn.Dropout2d(p=0.05)

        # image_size = 150 floor(((image size + (2 * padding) - kernel size) / stride) + 1)

        self.batch_norm = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=self.calc_pad(kernel_size=3))

        # floor(((150 + (2 * 1) - 3) / 2) + 1)

        # image_size = 150

        self.res_block1 = ResBlock(in_channels=64, out_channels=64, stride=1)

        self.res_block2 = ResBlock(in_channels=64, out_channels=128, stride=2)

        self.res_block3 = ResBlock(in_channels=128, out_channels=256, stride=2)

        self.res_block4 = ResBlock(in_channels=256, out_channels=512, stride=2)

        self.glob_avg_pool = nn.MaxPool2d(kernel_size=[8, 4])

        self.flatten = nn.Flatten()

        self.fully = nn.Linear(512, 256)

        self.selu2 = nn.SELU()

        self.batch_norm2 = nn.BatchNorm1d(256)

        self.dropout2 = nn.Dropout(p=0.05)

        self.fully2 = nn.Linear(256, n_classes)

    def calc_pad(self, kernel_size):
        # Attention: This does not apply for even kernel sizes! (Asymmetric padding would be needed then.)
        return int((kernel_size - 1) / 2)

    def forward(self, x):
        x = self.conv1(x)

        x = self.batch_norm(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.max_pool(x)

        x = self.res_block1(x)

        x = self.res_block2(x)

        x = self.res_block3(x)
        #
        x = self.res_block4(x)

        x = self.glob_avg_pool(x)

        # input should have size B x 512
        x = self.flatten(x)

        x = self.selu2(self.fully(x))

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = self.fully2(x)

        return x
