import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AF_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, n_pixel):
        super(AF_Block, self).__init__()
        self.conv_sc = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True)

        # condition attention
        self.fc_ca = nn.Linear(in_features=cond_dim, out_features=n_pixel, bias=True)
        self.conv_cond = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        t = int( (np.log2(in_ch) + 1) / 2)
        self.adapt_k = t if t % 2 else t + 1  # +1: make the k an odd number -> keep the dim unchange
        self.adapt_p = int((self.adapt_k - 1) / 2)
        self.conv_eca = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.adapt_k, stride=1, padding=self.adapt_p, bias=True)

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn_sc = nn.BatchNorm2d(out_ch)
        self.bn_af = nn.BatchNorm2d(out_ch)

    # short connection
    def shortcut(self, x):
        y = self.conv_sc(x)
        y = self.bn_sc(y)
        # y = F.leaky_relu(y, 0.2)
        y = F.relu(y)
        return y
    
    # channel attention
    def attention_ch(self, x):
        y = self.avg_pool(x)
        y = self.conv_eca(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        # y = F.leaky_relu(y, 0.2)
        y = F.relu(y)

        return x * y.expand_as(x)
    
    # condition attention
    def attention_cond(self, x, cond, n):
        y = self.fc_ca(cond)
        y = y.reshape(-1, 1, n, n)
        y = self.conv_cond(y)
        # y = F.leaky_relu(y, 0.2)
        y = F.relu(y)

        return x * y.expand_as(x)

    def forward(self, x, cond):
        n = x.shape[-1]
        out = self.attention_ch(x)
        out = self.attention_cond(out, cond, n)
        # out = self.attention_cond(x, cond, n)

        # out = self.attention_cond(x, cond, n)
        # out = self.attention_ch(out)
        out = self.conv(out)
        out = self.bn_af(out)
        # out = F.leaky_relu(out, 0.2)
        out = F.relu(out)
        out = self.shortcut(x) + out

        return  out

'''
Generator
nz: Size of z latent vector (i.e. size of generator input is batch_size * nz * 1 * 1)
ngf: Size of feature maps in generator
channels_img: Number of channels in the training images. For color images this is 3
'''
class NetG(nn.Module):
    def __init__(self, nz, ngf, channels_img = 3, embedding_dim = 300):
        super(NetG, self).__init__()

        self.ngf = ngf

        # self.fc = nn.Linear(embedding_dim+nz, ngf*8*4*4)
        self.fc = nn.Linear(embedding_dim+nz, ngf*8*4*4)
        
        # set the bias=False: it’s because the beta term in batch norm effectively adds a bias to each channel
        self.g_blocks = nn.ModuleList([
            # state size: ngf*8 * 4 * 4 -> ngf*8 * 8 * 8
            AF_Block(cond_dim=embedding_dim, in_ch=ngf*8, out_ch=ngf*4, n_pixel = 8*8),

            # state size: ngf*8 * 8 * 8 -> ngf*4 * 16 * 16
            AF_Block(cond_dim=embedding_dim, in_ch=ngf*4, out_ch=ngf*2, n_pixel = 16*16),

            # state size: ngf*4 * 16 * 16 -> ngf*2 * 32 * 32
            AF_Block(cond_dim=embedding_dim, in_ch=ngf*2, out_ch=ngf*1, n_pixel = 32*32),

            # state size: ngf*2 * 32 * 32 -> ngf*1 * 64 * 64
            AF_Block(cond_dim=embedding_dim, in_ch=ngf*1, out_ch=int(ngf*0.5), n_pixel = 64*64),
        ])

        self.to_rgb = nn.Sequential(
            nn.Conv2d(in_channels=int(ngf*0.5), out_channels=channels_img, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            )
    
    def forward(self, noise, caption_vec):
        # cap_features = self.cap_dim_reduce(caption_vec)
        cond = torch.cat((noise, caption_vec), dim=1)
        out = self.fc(cond)
        out = out.view(-1, 8*self.ngf, 4, 4)

        for af_block in self.g_blocks:
            out = F.interpolate(out, scale_factor=2)
            out = af_block(out, caption_vec)

        out = self.to_rgb(out)

        return out


'''
Discriminator
nz: Size of z latent vector (i.e. size of generator input is batch_size * nz * 1 * 1)
ndf: Size of feature maps in generator
channels_img: Number of channels in the training images. For color images this is 3
'''
class NetD(nn.Module):
    def __init__(self, nz, ndf, channels_img = 3, embedding_dim = 300):
        super(NetD, self).__init__()

        self.cap_dim_reduce = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=nz, bias=False),
            # nn.BatchNorm1d(nz),
            nn.ReLU()
        )
        
        # set the bias=False: it’s because the beta term in batch norm effectively adds a bias to each channel
        self.discriminator = nn.Sequential(
            # state size: channels_img * 64 * 64 -> ndf * 32 * 32
            nn.Conv2d(in_channels=channels_img, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),

            # state size: ndf * 32 * 32 -> ndf*2 * 16 * 16
            self.c_block(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False),

            # state size: ndf*2 * 16 * 16 -> ndf*4 * 8 * 8
            self.c_block(in_channels=ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1, bias=False),

            # state size: ndf*4 * 8 * 8 -> ndf*8 * 4 * 4
            self.c_block(in_channels=ndf*4, out_channels=ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
       
        )

        self.output = nn.Sequential(
            # state size: ndf*8 * 4 * 4 -> ndf*2 * 4 * 4
            self.c_block(in_channels=ndf*8+nz, out_channels=ndf*2, kernel_size=3, stride=1, padding=1, bias=False),

            # state size: ndf*2 * 4 * 4 -> 1 * 1 * 1
            nn.Conv2d(in_channels=ndf*2, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # nn.Tanh()
        )
    
    # Convolution + Batch-Normalization
    def c_block(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        _block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        return _block

    
    def forward(self, img, caption_vec):
        cap_features = self.cap_dim_reduce(caption_vec)
        feature_num = cap_features.shape[-1]
        cap_features = cap_features.reshape(-1, feature_num, 1, 1)  # batch_size * 100 -> batch_size * 100 * 1 * 1
        cap_features = torch.tile(cap_features, (1, 1, 4, 4))  # batch_size * 100 * 1 * 1 -> batch_size * 100 * 4 * 4
        d_out = self.discriminator(img)
        cond_out = torch.concat((d_out, cap_features), dim=1)  # batch_size * (512+100) * 4 * 4 
        out = self.output(cond_out)
        return out


# custom weights initialization called on netG and netD
def initialize(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)
