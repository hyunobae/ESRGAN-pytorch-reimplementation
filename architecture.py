import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, inchannel, gchannel=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inchannel+gchannel*0, gchannel, (3,3), (1,1), (1,1))
        self.conv2 = nn.Conv2d(inchannel+gchannel*1, gchannel, (3,3), (1,1), (1,1))
        self.conv3 = nn.Conv2d(inchannel+gchannel*2, gchannel,(3,3), (1,1), (1,1))
        self.conv4 = nn.Conv2d(inchannel+gchannel*3, gchannel, (3,3), (1,1), (1,1))
        self.conv5 = nn.Conv2d(inchannel+gchannel*4, inchannel, (3,3), (1,1), (1,1))

        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out1 = self.lrelu(self.conv1(x))
        out2 = self.lrelu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.lrelu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.lrelu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.conv5(torch.cat([x, out1, out2, out3, out4], 1))
        out = torch.mul(out5, 0.2)
        out = out + x

        return out


class RRDB(nn.Module):
    def __init__(self, inchannel, gchannel=32):
        super(RRDB, self).__init__()
        self.layer1 = DenseBlock(inchannel, gchannel)
        self.layer2 = DenseBlock(inchannel, gchannel)
        self.layer3 = DenseBlock(inchannel, gchannel)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.mul(out,0.2)
        return out+x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.RRDB = RRDB(64, 32)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64*4, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv3 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        out = self.RRDB(x)
        out = self.conv1(out)

        x = torch.add(x, out)
        x = self.upsample(x)# x2
        x = self.upsample(x)# x4
        x = self.lrelu(self.conv2(x))
        x = self.conv3(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator).__init__()
        self.conv0_0 = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(64, 64, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(64, affine=True)

        self.conv1_0 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(128, affine=True)
        self.conv1_1 = nn.Conv2d(128, 128, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(128, affine=True)

        self.conv2_0 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(256, affine=True)
        self.conv2_1 = nn.Conv2d(256, 256, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(256, affine=True)

        self.conv3_0 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(512, affine=True)
        self.conv3_1 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(512, affine=True)

        self.conv4_0 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(512, affine=True)
        self.conv4_1 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(512, affine=True)

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.linear1 = nn.Linear(512*4*4, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        out = self.lrelu(self.conv0_0(x))
        out = self.lrelu(self.bn0_1(self.conv0_1(out)))

        out = self.lrelu(self.bn1_0(self.conv1_0(out)))
        out = self.lrelu(self.bn1_1(self.conv1_1(out)))

        out = self.lrelu(self.bn2_0(self.conv2_0(out)))
        out = self.lrelu(self.bn2_1(self.conv2_1(out)))

        out = self.lrelu(self.bn3_0(self.conv3_0(out)))
        out = self.lrelu(self.bn3_1(self.conv3_1(out)))

        out = self.lrelu(self.bn4_0(self.conv4_0(out)))
        out = self.lrelu(self.bn4_1(self.conv4_1(out)))

        out = torch.flatten(out)
        out = self.lrelu(self.linear1(out))
        out = self.linear2(out)
        return out
