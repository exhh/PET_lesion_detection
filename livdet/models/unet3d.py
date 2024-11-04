import torch
import torch.nn as nn
import torch.nn.functional as F

from ..torch_utils import to_device
from ..torch_utils import Indexflow
from .models import register_model
from .encoding import Encoding3D
import numpy as np

__all__ = ['UNet3D', 'unet3d']

def passthrough(x, **kwargs):
    return x

def convAct(nchan):
    return nn.ELU(inplace=True)

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class Normalization3d(nn.Module):
    def __init__(self, inChans, norm_type = 'instance_norm'):
        super(Normalization3d, self).__init__()
        if norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm3d(inChans, affine=True)
        elif norm_type == 'batch_norm':
            self.norm = nn.BatchNorm3d(inChans, affine=True)
        elif norm_type == 'group_norm':
            pass
        elif norm_type == 'layer_norm':
            pass

    def forward(self, input):
        return self.norm(input)

class ConvBN(nn.Module):
    def __init__(self, nchan, inChans=None):
        super(ConvBN, self).__init__()
        if inChans is None:
            inChans = nchan
        self.act = convAct(nchan)
        self.conv = nn.Conv3d(inChans, nchan, kernel_size=3, padding=1)
        self.bn = Normalization3d(nchan)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out

def _make_nConv(nchan, depth):
    layers = []
    if depth >=0:
        for _ in range(depth):
            layers.append(ConvBN(nchan))
        return nn.Sequential(*layers)
    else:
        return passthrough

class InputTransition(nn.Module):
    def __init__(self,inputChans, outChans):
        self.outChans = outChans
        self.inputChans = inputChans
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(inputChans, outChans, kernel_size=3, padding=1)
        self.bn = Normalization3d(outChans)
        self.relu = convAct(outChans)

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = self.relu(out)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        #outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1, stride=2)
        self.bn1 = Normalization3d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(out+down)
        return out

def match_tensor(out, refer_shape):
    skipdep, skiprow, skipcol = refer_shape
    dep, row, col = out.size()[2], out.size()[3], out.size()[4]
    if skipdep >= dep:
        pad_dep = skipdep - dep
        front_pad_dep  = pad_dep // 2
        back_pad_dep = pad_dep - front_pad_dep
        out = F.pad(out, (0, 0, 0, 0, front_pad_dep, back_pad_dep))
    else:
        crop_dep = dep - skipdep
        front_crop_dep  = crop_dep // 2
        back_crop_dep = front_crop_dep + skipdep
        out = out[:,:,front_crop_dep:back_crop_dep,:,:]

    if skipcol >= col:
        pad_col = skipcol - col
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col
        out = F.pad(out, (left_pad_col, right_pad_col, 0, 0, 0, 0))
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:,:,left_crop_col:right_col]

    if skiprow >= row:
        pad_row = skiprow - row
        top_pad_row  = pad_row // 2
        bottom_pad_row = pad_row - top_pad_row
        out = F.pad(out, (0,0, top_pad_row,bottom_pad_row, 0, 0))
    else:
        crop_row = row - skiprow
        top_crop_row  = crop_row // 2
        bottom_row = top_crop_row + skiprow
        out = out[:,:,:,top_crop_row:bottom_row,:]
    return out

class UpConcat(nn.Module):
    def __init__(self, inChans, hidChans, outChans, nConvs, dropout=False,stride=2):
        super(UpConcat, self).__init__()
        self.upsam = nn.Upsample(scale_factor=stride)
        self.conv = nn.Conv3d(inChans, hidChans, kernel_size=3, padding=1, stride=1)
        self.bn1 = Normalization3d(hidChans)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = convAct(hidChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.conv(self.upsam(out))))
        out = match_tensor(out, skipxdo.size()[2:])

        xcat = torch.cat([out, skipxdo], 1)
        out  = self.ops(xcat)
        out  = self.relu2(out + xcat)
        return out

class UpConv(nn.Module):
    def __init__(self, inChans, outChans, nConvs,dropout=False, stride = 2):
        super(UpConv, self).__init__()
        self.upsam = nn.Upsample(scale_factor=stride)
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1, stride=1)
        self.bn1 = Normalization3d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()

    def forward(self, x, dest_size):
        '''
        dest_size should be (row, col)
        '''
        out = self.do1(x)
        out = self.relu1(self.bn1(self.up_conv(out)))
        out = match_tensor(out, dest_size)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans,outChans=1,hidChans=None):
        super(OutputTransition, self).__init__()
        self.hidChans = hidChans
        if self.hidChans is not None:
            self.conv1 = nn.Conv3d(inChans, hidChans, kernel_size=3, padding=1)
            self.bn1   = Normalization3d(hidChans)
            self.relu1 = convAct(hidChans)
            self.conv2 = nn.Conv3d(hidChans, outChans, kernel_size=1)
        else:
            self.conv2 = nn.Conv3d(inChans, outChans, kernel_size=1)


    def forward(self, x):
        if self.hidChans is not None:
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.conv2(out)
        else:
            out = self.conv2(x)
        return out

class SpatialEncodeBlock3D(nn.Module):
    def __init__(self, inChans, numCodewords, numClasses):
        super(SpatialEncodeBlock3D, self).__init__()
        self.encode = Encoding3D(inChans, numCodewords)
        self.conv = nn.Conv3d(numCodewords, 1, kernel_size=1)
        self.bn = nn.InstanceNorm3d(1)
        self.sig = nn.Sigmoid()
        self.relu2 = convAct(inChans)
        self.conv2 = nn.Conv3d(inChans, 1, kernel_size=1)
    def forward(self, x):
        out_encode, out_coef = self.encode(x)
        out_coef = self.sig(self.bn(self.conv(out_coef)))
        out = self.relu2(x * out_coef)
        out_encode = self.conv2(out)
        return out, out_encode

class SpaOriginalUNet3Dall(nn.Module):
    def __init__(self, num_class=1, K=16):
        super(SpaOriginalUNet3Dall, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.in_tr    = InputTransition(1, 32)
        self.down_tr1 = DownTransition(32, 64, 1)
        self.down_tr2 = DownTransition(64, 128, 2)
        self.down_tr3 = DownTransition(128, 256, 2, dropout=True)
        self.down_tr4 = DownTransition(256, 256, 2, dropout=True)

        self.encode4 = SpatialEncodeBlock3D(256, K, num_class)
        self.encode3 = SpatialEncodeBlock3D(256, K, num_class)
        self.encode2 = SpatialEncodeBlock3D(128, K, num_class)
        self.encode1 = SpatialEncodeBlock3D(64, K, num_class)
        self.upsample4 = nn.Upsample(scale_factor=16, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_tr4 = UpConcat(256, 256, 512, 2, dropout=True)
        self.up_tr3 = UpConcat(512, 128, 256, 2, dropout=True)
        self.up_tr2 = UpConcat(256, 64, 128, 1)
        self.up_tr1 = UpConcat(128, 32, 64, 1)
        self.out_tr = OutputTransition(64, num_class, 32)
        self.fuseweight = nn.Parameter(torch.Tensor(1,5,1,1,1), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std = 1./(5**(1.0/2))
        self.fuseweight.data.uniform_(0, std)

    def forward(self, x):
        x = to_device(x,self.device_id)
        out_in = self.in_tr(x)
        out_down1 = self.down_tr1(out_in)
        out_down2 = self.down_tr2(out_down1)
        out_down3 = self.down_tr3(out_down2)
        out_down4 = self.down_tr4(out_down3)

        out_atten4, out_regu4 = self.encode4(out_down4)
        out_atten3, out_regu3 = self.encode3(out_down3)
        out_atten2, out_regu2 = self.encode2(out_down2)
        out_atten1, out_regu1 = self.encode1(out_down1)
        out_regu4 = self.upsample4(out_regu4)
        out_regu4 = match_tensor(out_regu4, x.size()[2:])
        out_regu3 = self.upsample3(out_regu3)
        out_regu3 = match_tensor(out_regu3, x.size()[2:])
        out_regu2 = self.upsample2(out_regu2)
        out_regu2 = match_tensor(out_regu2, x.size()[2:])
        out_regu1 = self.upsample1(out_regu1)
        out_regu1 = match_tensor(out_regu1, x.size()[2:])

        out_up4 = self.up_tr4(out_atten4, out_atten3)
        out_up3 = self.up_tr3(out_up4, out_atten2)
        out_up2 = self.up_tr2(out_up3, out_atten1)
        out_up1 = self.up_tr1(out_up2, out_in)
        out = self.out_tr(out_up1)

        # fusion of prediction maps
        out_fuse = torch.cat([out, out_regu1, out_regu2, out_regu3, out_regu4], 1)
        out_fuse = torch.sum(out_fuse * self.fuseweight, dim=1, keepdim=True)
        return out_fuse, out, out_regu4, out_regu3, out_regu2, out_regu1

    def predict(self, batch_data, batch_size=None):
        self.eval()
        total_num = batch_data.shape[0]
        if batch_size is None or batch_size >= total_num:
            x = to_device(batch_data, self.device_id, False).float()
            out_fuse, out, out_regu4, out_regu3, out_regu2, out_regu1 = self.forward(x)
            return (out_fuse.cpu().data.numpy(), out.cpu().data.numpy(), \
                    out_regu4.cpu().data.numpy(), out_regu3.cpu().data.numpy(), \
                    out_regu2.cpu().data.numpy(), out_regu1.cpu().data.numpy())
        else:
            results_fuse, results = [], []
            results_regu4, results_regu3, results_regu2, results_regu1 = [], [], [], []
            for ind in Indexflow(total_num, batch_size, False):
                data = batch_data[ind]
                data = to_device(data, self.device_id, False).float()
                out_fuse, out, out_regu4, out_regu3, out_regu2, out_regu1 = self.forward(data)
                results.append(out.cpu().data.numpy())
                results_regu4.append(out_regu4.cpu().data.numpy())
                results_regu3.append(out_regu3.cpu().data.numpy())
                results_regu2.append(out_regu2.cpu().data.numpy())
                results_regu1.append(out_regu1.cpu().data.numpy())
                results_fuse.append(out_fuse.cpu().data.numpy())
            return (np.concatenate(results_fuse,axis=0), np.concatenate(results,axis=0), \
                    np.concatenate(results_regu4,axis=0), np.concatenate(results_regu3,axis=0), \
                    np.concatenate(results_regu2,axis=0), np.concatenate(results_regu1,axis=0))

@register_model('spaounet3dall')
def spaounet3dall(num_cls=1, num_codeword = 16, **kwargs):
    model = SpaOriginalUNet3Dall(num_class=num_cls, K = num_codeword)
    return model
