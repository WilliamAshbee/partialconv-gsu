###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class PartialConv2d(nn.Conv2d):#declare class partialConv2d
    def __init__(self, *args, **kwargs): #declare init function

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs: #set multichannel
            self.multi_channel = kwargs['multi_channel'] #set multichannel
            kwargs.pop('multi_channel') #remove multichannel
        else:
            self.multi_channel = False  #else set to default of false

        if 'return_mask' in kwargs: #return_mask
            self.return_mask = kwargs['return_mask']#set returnmask
            kwargs.pop('return_mask')#pop returnmask
        else:
            self.return_mask = False #set default for returnmask

        super(PartialConv2d, self).__init__(*args, **kwargs)#call parent constructor

        if self.multi_channel: #if self.multi_channel
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])#more channels for mask
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]) #fewer channels for mask
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None) #initialize last size
        self.update_mask = None #initialize update_mask
        self.mask_ratio = None #initialize mask_ratio

    def forward(self, input, mask_in=None): #define forward prop
        assert len(input.shape) == 4 #ensure input has 4 dimensions
        if mask_in is not None or self.last_size != tuple(input.shape): #if mask exists or if last size does not equal input size
            self.last_size = tuple(input.shape)

            with torch.no_grad(): #turn off required grad flags
                if self.weight_maskUpdater.type() != input.type(): #if types are unequal
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)  #adjust the type of weight_maskUpdater to match that of the input

                if mask_in is None: #no mask provided
                    # if mask is not provided, create a mask
                    if self.multi_channel: #if self.multi_channel is set
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input) #create mask for multiple channels
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input) #create mask for single channel
                else:
                    mask = mask_in #set mask to given mask
                    #how do i get here???
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output
