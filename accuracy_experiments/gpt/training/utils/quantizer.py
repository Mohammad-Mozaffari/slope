import torch


def find_point_location(x):
    """Finds the location of the point in a tensor."""
    max = torch.max(torch.abs(x.view(-1)))
    return torch.ceil(torch.log2(max))


def find_data_type(num_bits):
    """Finds the data type for a given number of bits."""
    if num_bits > 32:
        raise ValueError("Number of bits must be less than 32")
    elif num_bits > 16:
        return torch.int32
    elif num_bits > 8:
        return torch.int16
    else:
        return torch.uint8

# def quantize_weights(x, num_bits=8, bias=None):
#     """Quantizes a tensor."""
#     point_location = min(find_point_location(x), num_bits)
#     if bias is not None:
#         point_location = min(max(point_location, find_point_location(bias)), num_bits)
#     range = 2 ** point_location
#     x = x.clamp(-range, range - 1 / (2 ** (num_bits - 1)))
#     qx = torch.round(x / range * (2 ** (num_bits - 1) - 1))
#     qx = qx.to(data_type[num_bits])
#     if bias is not None:
#         qbias = torch.round(bias / range * (2 ** (num_bits - 1) - 1))
#         qbias = qbias.to(data_type[num_bits])
#         return qx, qbias, point_location
#     return qx, None, point_location


# def quantize_input(x, num_bits=8, point_location=None):
#     """Quantizes a tensor."""
#     return_point_location = False
#     if point_location is None:
#         point_location = min(find_point_location(x), num_bits)
#         return_point_location = True
#     range = 2 ** point_location
#     x = x.clamp(-range, range - 1 / (2 ** (num_bits - 1)))
#     qx = torch.round(x / range * (2 ** (num_bits - 1) - 1))
#     qx = qx.to(data_type[num_bits])
#     if return_point_location:
#         return qx, point_location
#     else:
#         return qx


# def dequantize(qx, point_location, num_bits=8):
#     """Dequantizes a tensor."""
#     if qx is None:
#         return None
#     x = qx.to(torch.float32) / (2 ** (num_bits - 1) - 1) * 2 ** point_location
#     return x

class QuantizedData():
    def __init__(self, num_bits=8, bit_mask_params=None):
        self.num_bits = num_bits
        self.zero_point = None
        self.scale = None
        self.bit_mask_params = bit_mask_params

    def dequantize(self, qx):
        """Dequantizes a tensor."""
        if qx is None:
            return None
        if self.bit_mask_params is not None:
            mask = torch.cumprod(torch.sigmoid(self.bit_mask_params - 8.0), dim=0)
            masked_qx = 0.0
            for i in range(self.num_bits):
                masked_qx += (qx % 2).to(torch.float32) * mask[-i] * 2 ** i
                qx = torch.div(qx, 2, rounding_mode='trunc')
            # print("MSE" , torch.mean((masked_qx - qx) ** 2))
            x = (masked_qx - self.zero_point.to(torch.float32)) / (2 ** self.num_bits - 1) * self.scale
        else:
            x = (qx.to(torch.float32) - self.zero_point.to(torch.float32)) / (2 ** self.num_bits - 1) * self.scale
        return x
        

class QuantizedParams(QuantizedData):
    def __init__(self, weight, num_bits=8, bias=None, bit_mask_params=None, save_original=False):
        super(QuantizedParams, self).__init__(num_bits=num_bits, bit_mask_params=bit_mask_params)
        self.quantize(weight, num_bits, bias)
        self.save_original = save_original
        if save_original:
            self.original_weight = weight
            self.original_bias = bias


    def quantize(self, weight, num_bits=8, bias=None):
        """Quantizes a tensor."""
        max = torch.max(torch.cat([weight.view(-1), bias.view(-1) if bias is not None else weight[0].view(-1)]))
        min = torch.min(torch.cat([weight.view(-1), bias.view(-1) if bias is not None else weight[0].view(-1)]))
        self.scale = max - min
        self.zero_point = (torch.round(-min / self.scale * (2 ** num_bits - 1))).to(torch.int32)
        self.weight = torch.round(weight / self.scale * (2 ** num_bits - 1) + self.zero_point).to(find_data_type(num_bits))
        self.bias = torch.round(bias / self.scale * (2 ** num_bits - 1) + self.zero_point).to(find_data_type(num_bits)) if bias is not None else None

    
    def requantize(self, new_num_bits):
        if self.save_original:
            self.quantize(self.original_weight, new_num_bits, self.original_bias)
        else:
            if new_num_bits > self.num_bits:
                raise ValueError("New number of bits must be less than the old number of bits.")
            self.weight = torch.round(self.weight / (2 ** (self.num_bits - new_num_bits))).to(find_data_type(new_num_bits))
            self.bias = torch.round(self.bias / (2 ** (self.num_bits - new_num_bits))).to(find_data_type(new_num_bits)) if self.bias is not None else None
            self.zero_point = torch.round(self.zero_point / (2 ** (self.num_bits - new_num_bits))).to(torch.int32)
            self.num_bits = new_num_bits
    

    def dequantize(self, param):
        if param == "weight":
            return super(QuantizedParams, self).dequantize(self.weight)
        elif param == "bias":
            return super(QuantizedParams, self).dequantize(self.bias)
        else:
            raise ValueError("Invalid parameter name.")


class QuantizedInput(QuantizedData):
    def __init__(self, val, num_bits=8, zero_point=None, scale=None):
        super(QuantizedInput, self).__init__(num_bits)
        self.zero_point = zero_point
        self.scale = scale
        self.quantize(val)

    def quantize(self, val):
        """Quantizes a tensor."""
        if self.zero_point is None or self.scale is None:
            max = torch.max(val.view(-1))
            min = torch.min(val.view(-1))
            self.scale = max - min
            self.zero_point = (torch.round(-min / self.scale * (2 ** self.num_bits - 1))).to(torch.int32)
        val = val.clamp(min, max)
        self.qval = torch.round(val / self.scale * (2 ** self.num_bits - 1) + self.zero_point).to(find_data_type(self.num_bits))

    
    def dequantize(self):
        return super(QuantizedInput, self).dequantize(self.qval)


class Quantizer(QuantizedData):
    def __init__(self, num_bits=8, status="learning", device='cuda'):
        super(Quantizer, self).__init__(num_bits)
        self.min = torch.tensor(torch.inf, device=device)
        self.max = torch.tensor(-torch.inf, device=device)
        self.status = status
        self.change_status_cnt = 10

    def quantize(self, val):
        """Quantizes a tensor."""
        if self.status != "evaluating":
            self.max = torch.max(torch.cat([val.view(-1), self.max.view(-1) if self.status == "learning" else val[0].view(-1)]))
            self.min = torch.min(torch.cat([val.view(-1), self.min.view(-1) if self.status == "learning" else val[0].view(-1)]))
            self.scale = self.max - self.min
            self.zero_point = (torch.round(-self.min / self.scale * (2 ** self.num_bits - 1))).to(torch.int32)
            self.change_status_cnt -= self.status == "learning"
            if self.change_status_cnt == 0:
                self.status = "evaluating"
        val = val.clamp(self.min, self.max)
        qval = torch.round(val / self.scale * (2 ** self.num_bits - 1) + self.zero_point).to(find_data_type(self.num_bits))
        return qval
    
    def dequantize(self, qval):
        return super(Quantizer, self).dequantize(qval)

    def requantize(self, num_bits):
        if num_bits > self.num_bits:
            raise ValueError("New number of bits must be less than the old number of bits.")
        if self.zero_point is not None:
            self.zero_point = torch.round(self.zero_point / (2 ** (self.num_bits - num_bits))).to(torch.int32)
            self.scale = self.scale * (2 ** (self.num_bits - num_bits))
        self.num_bits = num_bits