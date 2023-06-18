import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder,BasicEncoder4
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

#############################################################
class RCNCell(nn.Module):
    r"""abstract class for rcn cells. All rcn cells inheriting this class must
    implement the forward fuction.
    Inputs: x, hidden
        - **x** (batch, channel, height, width): tensor containing input features.
        - **hidden**: tensor or list of tensors that containing the hidden
            state for the precious time instance.
    Outputs: output, h
        - **output**: same as h
        - **h**: tensor or list of tensors containing the next hidden state
    """
    def forward(self, x, hidden):
        raise NotImplementedError("function forward(self, x, hidden) \
        is not implemented")

def Conv2dAndBN(use_bn, *param_list, **param_dict):
    conv = nn.Conv2d(*param_list, bias=not use_bn, **param_dict)
    if not use_bn:
        return conv
    else:
        bn = nn.BatchNorm2d(conv.out_channels)
        bn.weight.data.fill_(1)
        return nn.Sequential(conv, bn)

class GRURCNCellBase(RCNCell):

    def __init__(self, xz, hz, xr, hr, xh, rh):
        super(GRURCNCellBase, self).__init__()
        self._xz = xz
        self._hz = hz
        self._xr = xr
        self._hr = hr
        self._xh = xh
        self._rh = rh

    def forward(self, x, hidden):
        if hidden is None:
            h = torch.sigmoid(self._xz(x)) * torch.tanh(self._xh(x))
            return h, h
        else:
            z = torch.sigmoid(self._xz(x) + self._hz(hidden))
            r = torch.sigmoid(self._xr(x) + self._hr(hidden))
            h_ = torch.tanh(self._xh(x) + self._rh(r * hidden))
            h = (1 - z) * hidden + z * h_
            return h, h

class ConvGRURCNCell(RCNCell):
    r"""GRU-RCN cell using the conventional covolution as the inner operation.
    The convolutions on the hidden states must keep the size unchanged, so
    we do not provide the arguments to set the stride or padding of these
    convolutions, but fix stride=1 and padding=(kernel_size - 1) / 2).
    The convolutions on the inputs must make the outputs has the same size as
    hidden states. If the sizes of inputs and hidden states are not the same,
    users must provide the arguments (x_channels, x_kernel_size, x_stride,
    x_padding) to insure the outputs after the convolutions on inputs to be the
    same as the hidden states.
    Args: channels, kernel_size, ...
        - **channels**: the number of channels of hidden states. If x_channels
        is None, it means the channels of inputs and hidden states are the same.
        - **kernel_size**: size of the convolving kernel.
        - **x_channels**, **x_kernel_size**, **x_stride**, **x_padding**:
            parameters of the convolution operation on inputs.
            If None, inputs and hidden states have the same sizes, so the
            convolutions on them have no difference.
        - **batch_norm**: (bool) whether use batch norm or not
    Inputs: x (input), hidden
        - **x** (batch, channel, height, width): tensor containing input features.
        - **hidden** (batch, channel, height, width): tensor containing the hidden
            state for the precious time instance.
    Outputs: output, h
        - **output**: same as h
        - **h**: (batch, batch, channel, height, width): tensor containing the next
        hidden state
    """

    def __init__(self, channels, kernel_size, x_channels=None,
                 x_kernel_size=None, x_stride=None, x_padding=None, batch_norm=False):
        super(ConvGRURCNCell, self).__init__()

        x_channels = x_channels or channels
        x_kernel_size = x_kernel_size or kernel_size
        x_stride = x_stride or 1
        x_padding = x_padding or ((kernel_size - 1) // 2)

        hz = Conv2dAndBN(batch_norm, channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        hr = Conv2dAndBN(batch_norm, channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        rh = Conv2dAndBN(batch_norm, channels, channels, kernel_size, padding=(kernel_size - 1) // 2)

        xz = Conv2dAndBN(batch_norm, x_channels, channels, x_kernel_size, x_stride, x_padding)
        xr = Conv2dAndBN(batch_norm, x_channels, channels, x_kernel_size, x_stride, x_padding)
        xh = Conv2dAndBN(batch_norm, x_channels, channels, x_kernel_size, x_stride, x_padding)

        self._cell = GRURCNCellBase(xz, hz, xr, hr, xh, rh)

    def forward(self, x, hidden):
        return self._cell(x, hidden)
#############################################################################
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        # self.num_ff = para.future_frames
        # self.num_fb = para.past_frames
        # self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = 16
        self.model = nn.Sequential(
            nn.ConvTranspose2d(256, 4* self.n_feats, kernel_size=3, stride=2,padding=1, output_padding=1),
            nn.ConvTranspose2d(4 * self.n_feats, 2 * self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, 3, stride=1)
        )

    def forward(self, x):
        return self.model(x)

#############################################################################

class NoiseFlow(nn.Module):
    def __init__(self, args):
        super(NoiseFlow, self).__init__()
        self.args = args
        self.num_frames=args.num_frames
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder4(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.cell = ConvGRURCNCell(channels=256,kernel_size=3)
        self.recons = Reconstructor(self.args)
        self.device = torch.device('cuda')
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, x_list, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        for n in range(self.num_frames):
            x_list[:, n, :, :, :] = 2 * (x_list[:, n, :, :, :]/ 255.0) - 1.0
            x_list[:, n, :, :, :] = x_list[:, n, :, :, :].contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim


        # run the feature network
        fmap_list=[]
        with autocast(enabled=self.args.mixed_precision):
            for i in range(self.num_frames-1):
                fmap = self.fnet([x_list[:, i, :, :, :], x_list[:, i+1, :, :, :]])
                fmap = fmap.float()
                fmap_list.append(fmap)


        batch_size, n_feats, s_height, s_width = fmap_list[0].shape


        outputs, hs = [], []
        s = torch.zeros(batch_size, 256, s_height, s_width).to(self.device)


        for i in range(self.num_frames-1):
            h, s = self.cell(fmap_list[i], s)
            hs.append(h)

        fmap_r1,fmap_r2=torch.split(hs[-1], [1, 1], dim=0)
        #print("-------------------------------------fmap_r1:",fmap_r1.shape)

        clean_image= self.recons(fmap_r1)
        #print("-------------------------------------clean_image:",clean_image.shape)
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap_r1, fmap_r2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap_r1, fmap_r2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(x_list[:, int(self.num_frames/2), :, :, :])
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)


        coords0, coords1 = self.initialize_flow(x_list[:, int(self.num_frames/2), :, :, :])
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return clean_image, coords1 - coords0, flow_up
            
        return clean_image,flow_predictions
