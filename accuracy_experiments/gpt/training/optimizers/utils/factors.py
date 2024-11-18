import os
import torch
import torch.nn as nn
from torch import einsum
from torch.nn import Unfold
import torch.nn.functional as F


def try_contiguous(x):
    """
    :param x: input tensor
    """
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x

class ComputeI:
    """
    Compute the input tensor for the layer
    """
    @classmethod
    def compute_cov_a(cls, a, m):
        return cls.__call__(a, m)

    @classmethod
    def __call__(cls, a, m):
        if isinstance(m, nn.Linear):
            I = cls.linear(a, m)
            return I
        elif isinstance(m, nn.Conv2d):
            I = cls.conv2d(a, m)
            return I
        else:
            raise NotImplementedError

    @staticmethod
    def conv2d(input, m):
            f = Unfold(
                    kernel_size=m.kernel_size,
                    dilation=m.dilation,
                    padding=m.padding,
                    stride=m.stride)
            I = f(input)
            N, K, L = I.shape[0], I.shape[1], I.shape[2]
            M = m.out_channels
            m.param_shapes = [N, K, L, M]

            I = einsum('nkl->nk', I) # reduce sum over spatial dimension
            if m.bias is not None:
                return torch.cat([I / L, I.new(I.size(0), 1).fill_(1)], 1)
            return I / L

    @staticmethod
    def linear(input, m):
            if len(input.shape) == 3:
                input = input.reshape(-1, input.shape[-1])
            I = input
            N = I.shape[0]
            if m.bias is not None:
                return torch.cat([I, I.new(I.size(0), 1).fill_(1)], 1)
            return I


class ComputeG:
    """
    Compute the gradient tensor for the layer
    """
    @classmethod
    def compute_cov_g(cls, g, m):
        return cls.__call__(g, m)

    @classmethod
    def __call__(cls, g, m):
        if isinstance(m, nn.Linear):
            G, topk = cls.linear(g, m)
            return G, topk
        elif isinstance(m, nn.Conv2d):
            G, topk = cls.conv2d(g, m)
            return G, topk
        else:
            raise NotImplementedError

    @staticmethod
    def conv2d(g, m):
            n = g.shape[0]
            g_out_sc = n * g
            G = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)

            N, K, L, M = m.param_shapes
            G = einsum('nkl->nk', G) # reduce sum over spatial dimension
            topk = None
            return G / L, topk

    @staticmethod
    def linear(g, m):
            if len(g.shape) == 3:
                g = g.reshape(-1, g.shape[-1])
            n = g.shape[0]
            g_out_sc = n * g
            G = g_out_sc
            topk = None
            return G, topk


class ComputeCovA:
    """
    Compute the covariance of the input tensor for the layer
    """

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a = cls.conv2d(a, layer)
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            cov_a = None

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        # FIXME(CW): do we need to divide the output feature map's size?
        return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        if len(a.shape) == 3:
            a = a.reshape(-1, a.shape[-1])
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a.t() @ (a / batch_size)


class ComputeCovG:
    """
    Compute the covariance of the gradient tensor for the layer
    """
    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g = cls.linear(g, layer, batch_averaged)
        else:
            cov_g = None

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        if len(g.shape) == 3:
            g = g.reshape(-1, g.shape[-1])
        batch_size = g.size(0)

        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g


def sm_inverse(prev_inv, rank_1):
    """
    Sherman-Morrison formula for fast inverse update
    """
    tmp1 = (prev_inv @ rank_1)
    tmp2 = (rank_1.t() @ prev_inv)
    return prev_inv - 1 / (1 + tmp2 @ rank_1) * tmp1 @ tmp2
