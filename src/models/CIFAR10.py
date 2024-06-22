import torch
from torch import nn


class CIFAR300K(nn.Module):
    def __init__(self):
        super(CIFAR300K, self).__init__()

        self.conv1 = self._make_separable_conv(in_channels=3, out_channels=64)
        self.conv2 = self._make_separable_conv(in_channels=64, out_channels=64)

        self.conv3 = self._make_separable_conv(
            in_channels=64, out_channels=128, stride=2
        )
        self.conv4 = self._make_separable_conv(in_channels=128, out_channels=128)
        self.conv5 = self._make_separable_conv(in_channels=128, out_channels=128)

        self.conv6 = self._make_separable_conv(
            in_channels=128, out_channels=256, stride=2
        )
        self.conv7 = self._make_separable_conv(in_channels=256, out_channels=256)
        self.conv8 = self._make_separable_conv(in_channels=256, out_channels=256)

        self.conv9 = self._make_separable_conv(
            in_channels=256, out_channels=512, stride=2
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def _make_separable_conv(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.conv9(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class CIFAR900K(nn.Module):
    def __init__(self):
        super(CIFAR900K, self).__init__()

        self.conv1 = self._make_separable_conv(in_channels=3, out_channels=64)
        self.conv2 = self._make_separable_conv(in_channels=64, out_channels=64)

        self.conv3 = self._make_separable_conv(
            in_channels=64, out_channels=128, stride=2
        )
        self.conv4 = self._make_separable_conv(in_channels=128, out_channels=128)
        self.conv5 = self._make_separable_conv(in_channels=128, out_channels=128)

        self.conv6 = self._make_separable_conv(
            in_channels=128, out_channels=256, stride=2
        )
        self.conv7 = self._make_separable_conv(in_channels=256, out_channels=256)
        self.conv8 = self._make_separable_conv(in_channels=256, out_channels=256)

        self.conv9 = self._make_separable_conv(
            in_channels=256, out_channels=512, stride=2
        )
        self.conv10 = self._make_separable_conv(in_channels=512, out_channels=512)
        self.conv11 = self._make_separable_conv(in_channels=512, out_channels=512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def _make_separable_conv(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class CIFAR8M(nn.Module):
    def __init__(self):
        super(CIFAR8M, self).__init__()

        self.conv1 = self._make_conv(in_channels=3, out_channels=64)
        self.conv2 = self._make_conv(in_channels=64, out_channels=64)

        self.conv3 = self._make_conv(in_channels=64, out_channels=128, stride=2)
        self.conv4 = self._make_conv(in_channels=128, out_channels=128)
        self.conv5 = self._make_conv(in_channels=128, out_channels=128)

        self.conv6 = self._make_conv(in_channels=128, out_channels=256, stride=2)
        self.conv7 = self._make_conv(in_channels=256, out_channels=256)
        self.conv8 = self._make_conv(in_channels=256, out_channels=256)

        self.conv9 = self._make_conv(in_channels=256, out_channels=512, stride=2)
        self.conv10 = self._make_conv(in_channels=512, out_channels=512)
        self.conv11 = self._make_conv(in_channels=512, out_channels=512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def _make_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
