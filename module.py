from __future__ import division

from typing import Any

from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import (Add, AveragePooling2D, BatchNormalization,
                                     Conv2DTranspose, Dense, GlobalAveragePooling2D,
                                     SeparableConv2D, UpSampling2D)
from tensorflow.keras.regularizers import L1L2
from tensorflow_addons.activations import mish


def make_conv(input_layer, features, stride=1, decay=1e-2, transpose=False, dilation=1):
    layer = Conv2DTranspose if transpose else SeparableConv2D
    extra_params = {'kernel_initializer': 'orthogonal',
                    'kernel_regularizer': L1L2(decay)
                    } if transpose else {'pointwise_initializer': 'orthogonal',
                                         'depthwise_initializer': 'orthogonal',
                                         'depthwise_regularizer': L1L2(decay, decay),
                                         'pointwise_regularizer': L1L2(decay, decay)
                                         }
    return layer(filters=features,
                 kernel_size=1 + 2 * stride,
                 strides=stride,
                 padding=stride,
                 **extra_params,
                 dilation_rate=dilation)(input_layer)


def activate(input_layer):
    out = BatchNormalization()(input_layer)
    out = mish(out)
    return out


def double_conv(input_layer, features, stride=1, decay=1e-2, transpose=False):
    out = activate(make_conv(input_layer, features, 1, decay, False))
    out = make_conv(out, features, stride, decay, transpose)
    scale = (lambda x: x if stride == 1 else (
            UpSampling2D(stride) if transpose else AveragePooling2D(stride)))
    out = Add()([out, scale(input_layer)])
    out = activate(out)
    return out


def encoder(input, reuse: Any = None, name: Any = None):
    _ = reuse
    _ = name
    # 512 512 ???
    out = double_conv(input, 32, 2)  # 256 256 32
    out = double_conv(out, 64, 2)  # 128 128 64
    out = double_conv(out, 128, 2)  # 64 64 128
    out = double_conv(out, 256, 2)  # 32 32 256
    out = make_conv(out, 256, dilation=2)  # 32 32 256
    out = make_conv(out, 256, dilation=4)  # 32 32 256
    out = make_conv(out, 256, dilation=8)  # 32 32 256
    return out


def decoder(input, size1: Any = None, size2: Any = None, reuse: Any = None,
            name: Any = None):
    _ = size1
    _ = size2
    _ = reuse
    _ = name
    #                                                      X   Y   F
    #                                                      32  32 ???
    out = double_conv(input, 128, 2, transpose=True)  # .  64  64 128
    out = double_conv(out, 64, 2)  # .................... 128 128  64
    out = double_conv(out, 32)  # ....................... 128 128  32
    out = double_conv(out, 16)  # ....................... 128 128  16
    out = make_conv(out, 3)  # .......................... 128 128   3

    out = tanh(out)

    return out


def discriminator_red(input_layer, reuse: Any = None, name: Any = None):
    _ = reuse
    _ = name
    out = make_conv(input_layer, 16, 2)  # 128 128 16
    out = make_conv(out, 24, 2)  # 64 64 24
    out = make_conv(out, 36, 2)  # 32 32 36
    out = make_conv(out, 54, 2)  # 16 16 54
    out = make_conv(out, 81, 2)  # 8 8 81
    out = make_conv(out, 108, 2)  # 4 4 108
    out = GlobalAveragePooling2D()(out)
    out = Dense(2)(out)
    return out
