import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.cap_dim_reduce = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=nz, bias=False),
            nn.BatchNorm1d(nz),
            nn.Sigmoid()
        )

        # state size: nz -> ngf*8*4*4
        self.input = nn.Sequential(
            nn.Linear(in_features=2*nz, out_features=ngf*8*4*4, bias=False),
            nn.BatchNorm1d(ngf*8*4*4),
            nn.ReLU()
        )
        
        # set the bias=False: it’s because the beta term in batch norm effectively adds a bias to each channel
        self.convs = nn.ModuleList([
            # state size: ngf*8 * 4 * 4 -> ngf*4 * 8 * 8
            self.conv_block(in_channels=ngf*8, out_channels=ngf*4, kernel_size=3, stride=1, padding=1, bias=False),

            # state size: ngf*4 * 8 * 8 -> ngf*2 * 16 * 16
            self.conv_block(in_channels=ngf*4, out_channels=ngf*2, kernel_size=3, stride=1, padding=1, bias=False),

            # state size: ngf*2 * 16 * 16 -> ngf*1 * 32 * 32
            self.conv_block(in_channels=ngf*2, out_channels=ngf*1, kernel_size=3, stride=1, padding=1, bias=False),

            # state size: ngf*1 * 32 * 32 -> channels_img * 64 * 64
            self.to_rgb(in_channels=ngf*1, out_channels=channels_img, kernel_size=3, stride=1, padding=1, bias=False),
        ])

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        _block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return _block

    def to_rgb(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        _block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, bias=bias
            ),
            nn.Tanh()
        )
        return _block
    
    def forward(self, noise, caption_vec):
        cap_features = self.cap_dim_reduce(caption_vec)
        cond_input = torch.concat((noise, cap_features), dim=1)
        out = self.input(cond_input )
        out = out.reshape(-1, self.ngf*8, 4, 4)

        for block in self.convs:
            out = F.interpolate(out, scale_factor=2)  # 2n * 2n
            out = block(out)

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

            # state size: ndf * 36 * 36 -> ndf*2 * 18 * 18
            self.c_block(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False),

            # state size: ndf*2 * 18 * 18 -> ndf*4 * 9 * 9
            self.c_block(in_channels=ndf*2, out_channels=ndf*4, kernel_size=4, stride=2, padding=1, bias=False),

            # state size: ndf*4 * 9 * 9 -> ndf*8 * 4 * 4
            self.c_block(in_channels=ndf*4, out_channels=ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
       
        )

        self.output = nn.Sequential(
            # state size: ndf*8 * 4 * 4 -> ndf*2 * 4 * 4
            # nn.Conv2d(in_channels=ndf*8+nz, out_channels=ndf*2, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.LeakyReLU(0.2),
            self.c_block(in_channels=ndf*8+nz, out_channels=ndf*2, kernel_size=3, stride=1, padding=1, bias=False),
            # state size: ndf*2 * 4 * 4 -> 1 * 1 * 1
            nn.Conv2d(in_channels=ndf*2, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.flatten_output = nn.Sequential(
            nn.Linear(in_features=(512+100) * 4 * 4, out_features=1),
            nn.Sigmoid()
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
        # out = self.flatten_output(cond_out.reshape(-1, (512+100) * 4 * 4 ))
        return out


# custom weights initialization called on netG and netD
def initialize(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)