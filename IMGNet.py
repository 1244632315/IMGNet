import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self, kz=(13,3), down=['res', 'res', 'res'], \
                 up=['ffa', 'ffa', 'ffa']):
        super().__init__()

        # channel
        channel = 2

        # MDR
        self.mdr = MDR()

        # enhancement
        self.inc = DoubleConv(1, 32*channel)
        self.atten = SmearAtten()

        # encoder
        self.down1 = (Down(32*channel, 64*channel, kz, down[0]))
        self.down2 = (Down(64*channel, 128*channel, kz, down[1]))
        self.down3 = (Down(128*channel, 128*channel, kz, down[2]))

        # shortcut atten
        self.atten1 = ChannelAttention(32*channel)
        self.atten2 = ChannelAttention(64*channel)
        self.atten3 = ChannelAttention(128*channel)

        # decoder
        self.up1 = (UpAdd(128*channel, 64*channel, (3,3), up[0]))
        self.up2 = (UpAdd(64*channel, 32*channel, (3,3), up[1]))
        self.up3 = (UpAdd(32*channel, 32*channel, (3,3), up[2]))

        # out
        self.outc = OutConv(32*channel, 1)


    def forward(self, x):
        flag = self.mdr(x)
        if flag: x = x.transpose(2,3)

        x0 = self.inc(x)
        x1 = x0*self.atten(x)+x

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        y3 = self.up1(x4, self.atten3(x3))
        y2 = self.up2(y3, self.atten2(x2))
        y1 = self.up3(y2, self.atten1(x1))

        logits = self.outc(y1)
        if flag: logits = logits.transpose(2,3)
        return logits


class MDR(nn.Module):
    def __init__(self, kz=(7,3)) -> None:
        super().__init__()
        self.kh = torch.ones((1, kz[0]))
        self.kv = torch.ones((kz[1], 1))

    def erode(self, x, k):
        b, c, h, w = x.shape
        unfolded = F.unfold(x, kernel_size=k.shape, padding=(k.shape[0]//2, k.shape[1]//2))
        eroded = unfolded.min(dim=1).values
        eroded = eroded.view(b, c, h, w)
        return eroded
    
    def dilate(self, x, k):
        b, c, h, w = x.shape
        unfolded = F.unfold(x, kernel_size=k.shape, padding=(k.shape[0]//2, k.shape[1]//2))
        dilated = unfolded.max(dim=1).values
        dilated = dilated.view(b, c, h, w)
        return dilated
    
    def open(self, x, k):
        x = self.erode(x, k)
        x = self.dilate(x, k)
        return x
    
    def close(self, x, k):
        x = self.dilate(x, k)
        x = self.erode(x, k)
        return x
    
    def calphi(self, x):
        dil = self.dilate(x, self.kh)
        bi = (x>dil.mean(dim=(2,3), keepdim=True)) & (x==dil)
        op = self.open(bi.float(), self.kv)
        return op

    def forward(self, x):
        op1 = self.calphi(x)
        op2 = self.calphi(x.transpose(2,3))
        flag = op1.sum()<op2.sum()
        return flag


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kz=(3,3), module='douleconv', scale=2):
        super().__init__()
        module = getModule(module)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale),
            module(in_channels, out_channels, kz=kz)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpAdd(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kz=(3,3), module='douleconv', scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        module = getModule(module)        
        self.conv = module(in_channels, out_channels, kz=kz)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.add(x2, x1)
        return self.conv(x)


def getModule(name):
    if name == 'doubleconv':
        return DoubleConv
    elif name == 'res':
        return ResBlock
    elif name == 'ffa':
        return FFABlock
    else:
        raise("No related module!!!")


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kz=(3,3)):
        super().__init__()
        self.pconv1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.pconv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kz, padding=(kz[0]//2, kz[1]//2), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kz, padding=(kz[0]//2, kz[1]//2), bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pconv1(x)
        shortcut = x
        x = self.conv1(self.relu(x))
        x = shortcut + self.relu(x)

        x = self.pconv2(x)
        shortcut = x
        x = self.conv2(self.relu(x))
        x = shortcut + self.relu(x)
        
        return x


class FFABlock(nn.Module):
    def __init__(self, in_channels, out_channels, kz,):
        super(FFABlock, self).__init__()
        self.pconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.conv1=nn.Conv2d(out_channels, out_channels, kz, 1, (kz[0]//2, kz[1]//2))
        self.act1=nn.ReLU()
        self.conv2=nn.Conv2d(out_channels,out_channels,kz, 1, (kz[0]//2, kz[1]//2))
        self.calayer=CALayer(out_channels)
        self.palayer=PALayer(out_channels)
    def forward(self, x):
        x = self.act1(self.pconv(x))
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x 
        return res
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kz=(3,3)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kz, padding=(kz[0]//2, kz[1]//2), bias=False)
        conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kz, padding=(kz[0]//2, kz[1]//2), bias=False)
        self.double_conv = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
        )
    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
    
class SmearAtten(nn.Module):
    """colomn sum attention"""
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        col_avg = torch.mean(x, dim=2, keepdim=True)
        return self.sigmoid(col_avg)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)