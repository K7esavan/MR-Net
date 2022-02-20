import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg

from PretrainedNetwork.resnet import BasicBlock


def conv5x5(in_channels, out_channels, mode=None, sigmoid=False):
    ops = [nn.Conv2d(in_channels, out_channels, 5, padding=2),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(True)]
    if mode == "down":
        ops.insert(0, nn.MaxPool2d(2))
    elif mode == "up":
        ops.insert(0, nn.Upsample(scale_factor=2))
    if sigmoid:
        ops.pop(-1)
        ops.append(nn.Tanh())
    return nn.Sequential(*ops)


def conv3x3(in_channels, out_channels, stride=1, rate=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, dilation=rate),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(True))


def conv1x1(in_channels, out_channels, full_seq=True, zero_init=False):
    if full_seq:
        ops = [nn.Conv2d(in_channels, out_channels, 1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(True)]
        if zero_init:
            nn.init.constant_(ops[1].weight, 0.)
        return nn.Sequential(*ops)
    else:
        return nn.Conv2d(in_channels, out_channels, 1)


class NonlocalBlock(nn.Module):
    def __init__(self, in_channels, scale_factor, different_src=False):
        super(NonlocalBlock, self).__init__()
        self.src_attn_conv = conv1x1(in_channels, in_channels // scale_factor, full_seq=False)  # theta
        self.guide_attn_conv = conv1x1(in_channels, in_channels // scale_factor, full_seq=False)  # phi
        self.src_conv = conv1x1(in_channels, in_channels // scale_factor, full_seq=False)  # g
        self.output_conv = conv1x1(in_channels // scale_factor, in_channels, zero_init=True)
        self.different_src = different_src
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        if self.different_src:
            src, guide = input
            batch_size, num_channel, w, h = src.shape
            src_attn = self.src_attn_conv(src).view(batch_size, -1, w * h).permute(0, 2, 1)
            guide_attn = self.guide_attn_conv(guide).view(batch_size, -1, w * h)
            attn = self.softmax(torch.bmm(src_attm, guide_attn))
            src_proj = self.src_conv(src).view(batch_size, num_channel, w * h).permute(0, 2, 1)

            out = torch.bmm(attn, src_proj)
            out = slef.output_conv(out) + src
        else:
            batch_size, num_channel, w, h = input.shape
            src_attn = self.src_attn_conv(input).view(batch_size, -1, w * h).permute(0, 2, 1)
            guide_attn = self.guide_attn_conv(input).view(batch_size, -1, w * h)
            attn = self.softmax(torch.bmm(src_attn, guide_attn))
            src_proj = self.src_conv(input).view(batch_size, -1, w * h).permute(0, 2, 1)
            out = torch.bmm(attn, src_proj).permute(0, 2, 1).view(batch_size, -1, w, h)
            out = self.output_conv(out) + input

        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_nonlocal=True):
        super(Decoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 128, 'up')
        self.conv2 = conv5x5(128 * 2, 128)
        self.conv3 = conv5x5(128, 64, 'up')
        if use_nonlocal:
            self.nonlocal1 = NonlocalBlock(128, 4, False)
            self.nonlocal2 = NonlocalBlock(64, 4, False)
        self.conv4 = conv5x5(64 * 2 + 128, 64)
        self.conv5 = conv5x5(64, 32, 'up')
        self.conv6 = conv5x5(32 * 2 + 64, 32)
        self.conv7 = conv5x5(32, out_channels, "up", sigmoid=True)
        self.use_nonlocal = use_nonlocal

    def forward(self, input, skip, background_feat=None):
        if self.use_nonlocal:
            output = self.conv2(torch.cat([skip[0], self.nonlocal1(self.conv1(input))], 1))
            output = self.conv4(torch.cat([skip[1], self.nonlocal2(self.conv3(output)), background_feat[0]], 1))
        else:
            output = self.conv2(torch.cat([skip[0], self.conv1(input)], 1))
            output = self.conv4(torch.cat([skip[1], self.conv3(output), background_feat[0]], 1))
        output = self.conv6(torch.cat([skip[2], self.conv5(output), background_feat[1]], 1))
        '''
        output = self.conv1(torch.cat([skip[0], self.conv0(input)], 1))
        if self.use_nonlocal:
            output = self.nonlocal1(output)
        output = self.conv3(torch.cat([skip[1], self.conv2(output)], 1))
        if self.use_nonlocal:
            output = self.nonlocal2(output)
        output = self.conv5(torch.cat([skip[2], self.conv4(output), background_feat[0]], 1))
        output = self.conv7(torch.cat([self.conv6(output), background_feat[1]], 1))
        return self.conv8(output)
        '''
        return self.conv7(output)


class PoseEncoder(nn.Module):
    def __init__(self, in_channels):
        super(PoseEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32, "down")
        self.conv2 = conv5x5(32, 32)
        self.conv3 = conv5x5(32, 64, "down")
        self.conv4 = conv5x5(64, 64)
        self.conv5 = conv5x5(64, 128, "down")
        self.conv6 = conv5x5(128, 128)
        self.conv7 = conv5x5(128, 256, "down")
        self.conv8 = conv5x5(256, 256)

    def forward(self, input):
        output = []
        output.append(self.conv1(input))
        output.append(self.conv2(output[-1]))
        output.append(self.conv3(output[-1]))
        output.append(self.conv4(output[-1]))
        output.append(self.conv5(output[-1]))
        output.append(self.conv6(output[-1]))
        output.append(self.conv7(output[-1]))
        output.append(self.conv8(output[-1]))
        return output[-1], [output[5], output[3], output[0]], [output[5], output[3], output[1]]


class ForeGroundEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ForeGroundEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32, "down")
        self.conv2 = conv5x5(32, 32)
        self.conv3 = conv5x5(32, 64, "down")
        self.conv4 = conv5x5(64, 64)
        self.conv5 = conv5x5(64, 128, "down")
        self.conv6 = conv5x5(128, 128)
        self.conv7 = conv5x5(128, 256, "down")
        self.conv8 = conv5x5(256, 256)

    def forward(self, input, lateral):
        output = []
        output.append(self.conv1(input))
        output.append(self.conv2(output[-1] + lateral[-1]))
        output.append(self.conv3(output[-1]))
        output.append(self.conv4(output[-1]))
        output.append(self.conv5(output[-1] + lateral[-2]))
        output.append(self.conv6(output[-1]))
        output.append(self.conv7(output[-1] + lateral[-3]))
        output.append(self.conv8(output[-1]))
        return output[-1], [output[5], output[3], output[1]]


class MainEncoder(nn.Module):
    def __init__(self, pose_channels, foreground_channels):
        super(MainEncoder, self).__init__()
        self.pose_enc = PoseEncoder(pose_channels)
        self.fore_enc = ForeGroundEncoder(foreground_channels)
        # self.conv = conv1x1(512, 256)
        self.conv = BasicBlock(256, 256, downsample=None)

    def forward(self, pose, foreground):
        pose_feat, lateral, pose_skip = self.pose_enc(pose)
        fore_feat, fore_skip = self.fore_enc(foreground, lateral)
        skip = list(map(lambda x: list(x)[0] + list(x)[1], zip(pose_skip, fore_skip)))
        feat = pose_feat + fore_feat
        feat = self.conv(feat)
        return feat, skip


class BackGroundEncoder(nn.Module):
    def __init__(self, in_channels, input_size, dataset='market1501'):
        super(BackGroundEncoder, self).__init__()
        self.conv1 = conv5x5(in_channels, 32)
        self.conv2 = conv5x5(32, 64, 'down')
        self.di_conv1 = conv3x3(64, 64, rate=2)
        self.di_conv2 = conv3x3(64, 64, rate=4)
        self.di_conv3 = conv3x3(64, 64, rate=8)
        '''
        self.conv3_1 = conv5x5(64, 64)
        '''
        self.conv3 = conv5x5(64, 128, 'down')
        size = (32, 16) if dataset == 'market1501' else (32, 32)
        self.upsample = nn.Upsample(size, mode='bilinear')

    def forward(self, input):
        output = []
        x = self.conv1(input)
        x = self.conv2(x)
        output.insert(0, x)
        x = self.di_conv1(x)
        x = self.di_conv2(x)
        x = self.di_conv3(x)
        x = self.conv3(x)
        x = self.upsample(x)
        '''
        x = self.conv3_1(x)
        x = self.conv3(x)
        '''
        output.insert(0, x)
        # return x
        return output


class Pose_to_Image(nn.Module):
    def __init__(self, pose_channels=3, img_channels=3, input_size=128, recurrent=1, use_nonlocal=True, dataset=None):
        super(Pose_to_Image, self).__init__()
        self.pose_channels = pose_channels
        self.img_channels = img_channels
        self.encoder_conv = MainEncoder(pose_channels, img_channels)
        self.decoder_conv = Decoder(256, 3, use_nonlocal=use_nonlocal)
        self.background_encoder = BackGroundEncoder(img_channels + 1, input_size, dataset=dataset)
        self.recurrent = recurrent

    def forward(self, input):
        pose_imgs, human_imgs, background_imgs, masks = input
        # pose_human_imgs = torch.cat([pose_imgs, human_imgs], 1)
        foreground_feat, skip = self.encoder_conv(pose_imgs, human_imgs)
        outputs = []
        for i in range(self.recurrent):
            background_feat = self.background_encoder(torch.cat([background_imgs, masks], 1))
            # output, skip = self.encoder_conv(pose_human_imgs)
            output = self.decoder_conv(foreground_feat, skip, background_feat)
            background_imgs = output
            outputs.append(output)
        return output, outputs


def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):  # pytorch version use for loop !!!
    assert rois.dim() == 2
    assert rois.size(1) == 5
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)
    rois[:, 1:] = rois[:, 1:].clone().mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1).clone()[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        output.append(F.adaptive_max_pool2d(im, size))

    output = torch.cat(output, 0)
    return output


class Perceptual(nn.Module):
    def __init__(self, mapping, roi_size):
        super(Perceptual, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layers_mapping = mapping
        self.roi_size = roi_size

    def forward(self, x, target_bbox=None, use_roi=False, size=(7, 7)):
        initial_size = x.size()
        image_w = initial_size[2]
        output = {}
        masked_output = {}
        roi_cnt = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layers_mapping:
                if use_roi:
                    spatial_scale = x.shape[2] / image_w
                    masked_output[self.layers_mapping[name]] = roi_pooling(x, target_bbox, size=self.roi_size[roi_cnt],
                                                                           spatial_scale=spatial_scale)
                    roi_cnt += 1
                output[self.layers_mapping[name]] = x
        return output, masked_output
