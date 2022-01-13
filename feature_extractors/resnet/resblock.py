import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.out_channels = out_channels

        self.stride = stride

        # All Conv2D layers within a ResBlock have a filter size of 3
        kernel_size = 3

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)

        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)

        self.batchnorm3 = nn.BatchNorm2d(out_channels)

        self.convSkipInput = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1, stride=stride)

        self.batchnormInput = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        output = F.relu(self.batchnorm1(self.conv1(input)))

        output = F.relu(self.batchnorm2(self.conv2(output)))

        output = F.relu(self.batchnorm3(self.conv3(output)))

        input = self.convSkipInput(input)

        output = output + self.batchnormInput(input)

        return output

if __name__ == "__main__":
    resBlock = ResBlock(64, 128, 1)
    resBlock.forward(torch.ones((1, 64, 300, 300)))
