import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models.builder import build_loss

from mmcv.utils import Registry, build_from_cfg
from .ADG import ADG

DepthNet = Registry('depthnet')


def build_depthnet(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, DepthNet, default_args)



@DepthNet.register_module()
class AdaptiveDepthNetV2(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels, num_params1=18,patch_size=4,shape=None,
                 num_params2=6, with_depth_correction=False, with_context_encoder=False, fix_alpha=None,
                 with_pgd=False, with_adptive_bins = False, max_val=51, min_val=1, n_bins=64):
        super(AdaptiveDepthNetV2, self).__init__()
        self.in_channels = in_channels
        # self.context_channels = context_channels
        self.depth_channels = depth_channels
        self.mid_channels = mid_channels
        self.fix_alpha = fix_alpha
        self.reduce_conv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        )
        self.n_bins= n_bins
        self.ADG = ADG(mid_channels,patch_size=patch_size, shape=shape, dim_out=self.n_bins)
        self.max_val = max_val
        self.min_val = min_val
        self.with_adptive_bins = with_adptive_bins
        self.conv_out = nn.Sequential(nn.Conv2d(128, self.n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.bn2 = nn.BatchNorm1d(num_params2) 
        self.depth_mlp = Mlp(num_params2, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        if self.with_adptive_bins:
            self.fuse_gamma = nn.Parameter(torch.tensor(10e-5))
        if with_depth_correction: # True
            self.depth_stem = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                ASPP(mid_channels, mid_channels),
            )
            self.depth_prob_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    im2col_step=128,
                )),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, depth_channels,
                        kernel_size=1, stride=1, padding=0)
            )
        else:
            self.depth_stem = torch.nn.Identity()
            if not with_adptive_bins:
                self.depth_prob_conv = nn.Conv2d(
                    mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)

        self.with_pgd = with_pgd
        if self.with_pgd:
            if self.fix_alpha is not None:
                self.fuse_lambda = torch.tensor(self.fix_alpha).cuda()
            else:
                self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))

            self.depth_direct_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    im2col_step=128,
                )),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0) # 输出直接为1通道
            )

    def forward(self, x, intrinsics, extrinsics):
        """
        B, N_view = intrinsics.shape[:2]
        intrinsics = intrinsics[..., :2, :]  # 6
        extrinsics = extrinsics[..., :3, :]  # 12
        camera_params = torch.cat(
            [intrinsics.view(B * N_view, -1), extrinsics.view(B * N_view, -1)], dim=-1) 


        x = self.reduce_conv(x) # 

        # down_sample_x = F.interpolate(x,scale_factor=0.5, mode='bilinear') 

        mlp_input = self.bn2(intrinsics.view(B * N_view, -1)) # [bs * N_view, 6]
    
        depth_se = self.depth_mlp(mlp_input)[..., None, None] # MLP [bs*n, 256,1,1]

        depth = self.depth_se(x, depth_se) 
       
        depth_stem = self.depth_stem(depth) #
        if self.with_adptive_bins:
            depth_adptive, bin_edges = self.adptive_bins_layer(depth_stem) #
            depth_prob = self.depth_prob_conv(depth_stem) # 
            depth_direct = self.depth_direct_conv(depth_stem) #
            return bin_edges, depth_adptive, depth_prob, x, depth_direct
  
    
    def adptive_bins_layer(self, depth_stem):
        bin_widths_normed, range_attention_maps = self.ADG(depth_stem) # [bs*n,256] [bs*n,128,20,50 ]
        range_attention_maps = self.conv_out(range_attention_maps) # [6,100,20,50]
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val) #
        bin_edges = torch.cumsum(bin_widths, dim=1) # 
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:]) 
        n, dout = centers.size() #
        centers = centers.view(n, dout, 1, 1) # 
        pred = torch.sum(range_attention_maps * centers, dim=1, keepdim=True) # [6,100,20,50] [6,100,1,1] -> [6,1,20,50]
        return  pred, bin_edges
