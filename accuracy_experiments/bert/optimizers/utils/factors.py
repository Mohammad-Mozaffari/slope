import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from torch.nn import Unfold


class ComputeI:
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
    def linear(input, m, enable_topk=False):
            if len(input.shape) == 3:
                input = input.reshape(-1, input.shape[-1])
            I = input
            N = I.shape[0]
            if m.bias is not None:
                return torch.cat([I, I.new(I.size(0), 1).fill_(1)], 1)
            return I


class ComputeG:
    @classmethod
    def compute_cov_g(cls, g, m, enable_topk=False):
        return cls.__call__(g, m, enable_topk=False)

    @classmethod
    def __call__(cls, g, m, enable_topk=False):
        if isinstance(m, nn.Linear):
            G, topk = cls.linear(g, m, enable_topk)
            return G, topk
        elif isinstance(m, nn.Conv2d):
            G, topk = cls.conv2d(g, m, enable_topk)
            return G, topk
        else:
            raise NotImplementedError

    @staticmethod
    def conv2d(g, m, enable_topk=False):
            n = g.shape[0]
            g_out_sc = n * g
            G = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)

            N, K, L, M = m.param_shapes
            G = einsum('nkl->nk', G) # reduce sum over spatial dimension
            topk = None
            # if enable_topk:
            #     norm = torch.norm(G, p=1, dim=1)
            #     k = int(N / 10)
            #     topk = torch.topk(norm, k).indices
            # if topk is not None:
            #     G = G[topk, :]
            return G / L, topk

    @staticmethod
    def linear(g, m, enable_topk=False):
            if len(g.shape) == 3:
                g = g.reshape(-1, g.shape[-1])
            n = g.shape[0]
            g_out_sc = n * g
            G = g_out_sc
            topk = None
            # if enable_topk:
            #     norm = torch.norm(G, p=1, dim=1)
            #     k = int(n / 10)
            #     topk = torch.topk(norm, k).indices
            # if topk is not None:
            #     G = G[topk, :]
            return G, topk
