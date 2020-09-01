#from https://github.com/ajbrock/BigGAN-PyTorch (MIT license)
# some modifications in class Generator and G_D
# new class "Unet_Discriminator" based on original class "Discriminator"
import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import layers

import utils
import copy
from matplotlib import pyplot as plt


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}

    arch[256] = {'in_channels' :    [ch * item for item in [16, 16, 8, 8, 4, 2]],
                             'out_channels' : [ch * item for item in [16,    8, 8, 4, 2, 1]],
                             'upsample' : [True] * 6,
                             'resolution' : [8, 16, 32, 64, 128, 256],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,9)}}
    arch[128] = {'in_channels' :    [ch * item for item in [16, 16, 8, 4, 2]],
                             'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
                             'upsample' : [True] * 5,
                             'resolution' : [8, 16, 32, 64, 128],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,8)}}
    return arch

class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
                             G_kernel_size=3, G_attn='64', n_classes=1000,
                             num_G_SVs=1, num_G_SV_itrs=1,
                             G_shared=True, shared_dim=0, hier=False,
                             cross_replica=False, mybn=False,
                             G_activation=nn.ReLU(inplace=False),
                             G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                             BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                             G_init='ortho', skip_init=False, no_optim=False,
                             G_param='SN', norm_style='bn',
                             **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]

        self.unconditional = kwargs["unconditional"]

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)

            if not self.unconditional:
                self.dim_z = self.z_chunk_size *    self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding

        if self.unconditional:
            bn_linear = nn.Linear
            input_size =  self.dim_z  + (self.shared_dim if self.G_shared else 0 )
        else:
            bn_linear = (functools.partial(self.which_linear, bias = False) if self.G_shared
                                     else self.which_embedding)

            input_size = (self.shared_dim + self.z_chunk_size if self.G_shared
                                    else self.n_classes)
        self.which_bn = functools.partial(layers.ccbn,
                                                    which_linear=bn_linear,
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn,
                                                    input_size=input_size,
                                                    norm_style=self.norm_style,
                                                    eps=self.BN_eps,
                                                    self_modulation = self.unconditional)


        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
                                        else layers.identity())
        # First linear layer
        if self.unconditional:
            self.linear = self.which_linear(self.dim_z, self.arch['in_channels'][0] * (self.bottom_width **2))
        else:
            self.linear = self.which_linear(self.dim_z // self.num_slots,
                                                                        self.arch['in_channels'][0] * (self.bottom_width **2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):


            self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                                     out_channels=self.arch['out_channels'][index],
                                                     which_conv=self.which_conv,
						                             which_bn=self.which_bn,
                                                     activation=self.activation,
                                                     upsample=(functools.partial(F.interpolate, scale_factor=2)
                                                                         if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                                                                cross_replica=self.cross_replica,
                                                                                                mybn=self.mybn),
                                                                        self.activation,
                                                                        self.which_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print('Using fp16 adam in G...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                                     betas=(self.B1, self.B2), weight_decay=0,
                                                     eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                                     betas=(self.B1, self.B2), weight_decay=0,
                                                     eps=self.adam_eps)

        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y ):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            # faces
            if self.unconditional:
                ys = [z for _ in range(self.num_slots)]
            else:
                zs = torch.split(z, self.z_chunk_size, 1)
                z = zs[0]

                ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            if self.unconditional:
                ys = [None] * len(self.blocks)
            else:
                ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels' :    [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                             'downsample' : [True] * 6 + [False],
                             'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,8)}}
    arch[128] = {'in_channels' :    [3] + [ch*item for item in [1, 2, 4, 8, 16]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
                             'downsample' : [True] * 5 + [False],
                             'resolution' : [64, 32, 16, 8, 4, 4],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,8)}}
    return arch

def D_unet_arch(ch=64, attention='64',ksize='333333', dilation='111111',out_channel_multiplier=1):
    arch = {}

    n = 2

    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128]= {'in_channels' :       [3] + [ch*item for item in       [1, 2, 4, 8, 16, 8*n, 4*2, 2*2, 1*2,1]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 8,   4,   2,    1,  1]],
                             'downsample' : [True]*5 + [False]*5,
                             'upsample':    [False]*5+ [True] *5,
                             'resolution' : [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,11)}}


    arch[256] = {'in_channels' :            [3] + [ch*item for item in [1, 2, 4, 8, 8, 16, 8*2, 8*2, 4*2, 2*2, 1*2  , 1         ]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 8,   8,   4,   2,   1,   1          ]],
                             'downsample' : [True] *6 + [False]*6 ,
                             'upsample':    [False]*6 + [True] *6,
                             'resolution' : [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,13)}}



    return arch


class Unet_Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                             D_kernel_size=3, D_attn='64', n_classes=1000,
                             num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                             D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                             SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                             D_init='ortho', skip_init=False, D_param='SN', decoder_skip_connection = True, **kwargs):
        super(Unet_Discriminator, self).__init__()


        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16



        if self.resolution==128:
            self.save_features = [0,1,2,3,4]
        elif self.resolution==256:
            self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention , out_channel_multiplier = self.out_channel_multiplier  )[resolution]

        self.unconditional = kwargs["unconditional"]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)

            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                            eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.D_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            elif self.arch["upsample"][index]:
                upsample_function = (functools.partial(F.interpolate, scale_factor=2, mode="nearest") #mode=nearest is default
                                    if self.arch['upsample'][index] else None)

                self.blocks += [[layers.GBlock2(in_channels=self.arch['in_channels'][index],
                                                         out_channels=self.arch['out_channels'][index],
                                                         which_conv=self.which_conv,
                                                         #which_bn=self.which_bn,
                                                         activation=self.activation,
                                                         upsample= upsample_function, skip_connection = True )]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition: #index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                                                         self.which_conv)]


        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])


        last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier,1,kernel_size=1)
        self.blocks.append(last_layer)
        #
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

        self.linear_middle = self.which_linear(16*self.ch, output_dim)
        # Embedding for projection discrimination
        #if not kwargs["agnostic_unet"] and not kwargs["unconditional"]:
        #    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1]+extra)
        if not kwargs["unconditional"]:
            self.embed_middle = self.which_embedding(self.n_classes, 16*self.ch)
            self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        ###
        print("_____params______")
        for name, param in self.named_parameters():
            print(name, param.size())

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)



    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index==6 :
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==7:
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==8:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[1]),dim=1)

            if self.resolution == 256:
                if index==7:
                    h = torch.cat((h,residual_features[5]),dim=1)
                elif index==8:
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==10:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==11:
                    h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index==self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(self.activation(h), [2, 3])
                # Get initial class-unconditional output
                bottleneck_out = self.linear_middle(h_)
                # Get projection of final featureset onto class vectors and add to evidence
                if self.unconditional:
                    projection = 0
                else:
                    # this is the bottleneck classifier c
                    emb_mid = self.embed_middle(y)
                    projection = torch.sum(emb_mid * h_, 1, keepdim=True)
                bottleneck_out = bottleneck_out + projection

        out = self.blocks[-1](h)

        if self.unconditional:
            proj = 0
        else:
            emb = self.embed(y)
            emb = emb.view(emb.size(0),emb.size(1),1,1).expand_as(h)
            proj = torch.sum(emb * h, 1, keepdim=True)
            ################
        out = out + proj

        out = out.view(out.size(0),1,self.resolution,self.resolution)

        return out, bottleneck_out


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
    def __init__(self, G, D, config):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

        self.config = config

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
                            split_D=False, dw1=[],dw2=[], reference_x = None, mixup = False, mixup_only = False, target_map=None):

        if mixup:
            gy = dy
            #why? so the mixup samples consist of same class

        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):

            G_z = self.G(z, self.G.shared(gy))
            # Cast as necessary
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()

        if mixup:
            initial_x_size = x.size(0)

            mixed = target_map*x+(1-target_map)*G_z
            mixed_y = dy


        if not mixup_only:
            # we get here in the cutmix cons extra case
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            dmap = torch.tensor([])
            if mixup:
                #we get here in the cutmix  "consistency loss and augmentation" case, if "mixup" is true for the current round (depends on p mixup)
                D_input = torch.cat([D_input, mixed], 0)
                if self.config["dataset"]!="coco_animals":
                    D_class = torch.cat([D_class.float(), mixed_y.float()], 0)
                else:
                    D_class = torch.cat([D_class.long(), mixed_y.long()], 0)
        else:
            #not reached in cutmix "consistency loss and augmentation"
            D_input = mixed
            D_class = mixed_y
            dmap = torch.tensor([])

            del G_z
            del x
            G_z = None
            x = None

        D_out, D_middle = self.D(D_input, D_class)

        del D_input
        del D_class


        if x is not None:

            if not mixup:
                out = torch.split(D_out, [G_z.shape[0], x.shape[0]])     # D_fake, D_real
            else:
                out = torch.split(D_out, [G_z.shape[0], x.shape[0], mixed.shape[0]])  # D_fake, D_real, D_mixed
            out = out + (G_z,)
            if mixup:
                out = out + (mixed,)

            if not mixup:
                D_middle =  torch.split(D_middle, [G_z.shape[0], x.shape[0]])     # D_middle_fake, D_middle_real
            else:
                D_middle =  torch.split(D_middle, [G_z.shape[0], x.shape[0] , mixed.shape[0]])
            out = out + D_middle
            ###return target map as well
            if mixup:
                out = out + (target_map,)

            return out


        else:
            #in mixup# you arrive here
            out = (D_out,)

            if return_G_z:
                out = out + (G_z,)
            if mixup_only:
                out = out + (mixed,)

            out =  out + (D_middle,)
            ##return target map as well
            if mixup:
                out = out + (target_map,)

            return out
