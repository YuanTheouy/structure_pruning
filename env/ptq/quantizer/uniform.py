# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):

    def __init__(self, scale, zero, bit, module_type):
        super(UniformQuantizer, self).__init__(bit, module_type)
        self.scale = scale
        self.zero_point = zero

    # def update_quantization_params(self, *args, **kwargs):
    #     self.scale, self.zero_point = self.observer.get_quantization_params(
    #         *args, **kwargs)

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        # range_shape = self.get_reshape_range(inputs)
        # scale = scale.reshape(range_shape)
        # zero_point = zero_point.reshape(range_shape)
        outputs = inputs * scale + zero_point
        outputs = outputs.round().clamp(-2**(self.bit-1),2**(self.bit-1))
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        # range_shape = self.get_reshape_range(inputs)
        # scale = scale.reshape(range_shape)
        # zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) / scale
        return outputs
