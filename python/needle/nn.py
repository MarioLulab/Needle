"""The module.
"""
from collections import OrderedDict
from typing import Any, Callable, Iterator, List, Optional, Set, Tuple, Union

import needle
import needle as ndl
import needle.init as init
import numpy as np
from needle import ops
from needle.autograd import Tensor
import math

import operator
from functools import reduce
def prod(x):
    return reduce(operator.mul, x, 1)


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        
        self.device = device
        self.dtype = dtype
        self.b = bias
        self.weight = Parameter(
            needle.init.kaiming_uniform(fan_in=in_features, fan_out=out_features, shape=(in_features, out_features)),
            device=self.device,
            dtype=self.dtype) # (in_features, out_features)
        self.bias = Parameter(
            needle.init.kaiming_uniform(fan_in=out_features, fan_out=1, shape=(out_features,)),
            device=self.device,
            dtype=self.dtype) # (out_features,)
        

    def forward(self, X: Tensor) -> Tensor:
        
        # X : (batch_size, in_features)
        # X = needle.matmul(X, self.weight.transpose())
        X = needle.matmul(X, self.weight)
        if self.b:
            X = X + self.bias.broadcast_to((X.shape[0], self.out_features))
        return X
        


class Flatten(Module):
    def forward(self, X):
        
        tail = prod(X.shape[1:])
        # return X.reshape( (X.shape[0], -1) )
        return X.reshape( (X.shape[0], tail) )
        


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        
        return needle.relu(x)
        


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        
        return needle.tanh(x)
        


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        
        return ndl.sigmoid(x)
        


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def __getitem__(self, idx):
        return self.modules[idx]

    def __setitem__(self, idx, module : Module):
        self.modules[idx] = module
    
    def __len__(self):
        return len(self.modules)

    def forward(self, x: Tensor) -> Tensor:
        
        for module in self.modules:
            x = module(x)
        return x
        


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        
        # logits : with shape (batch_size, num_cls)
        # y  : with shape (batch_size,)
        num_cls = logits.shape[-1]
        y_one_hot = init.one_hot(num_cls, y, device=y.device, dtype=y.dtype, requires_grad=False)    # (batch_size, num_cls)
        softmax_loss_ = ops.logsumexp(logits, axes=(1,)) - ops.summation(logits * y_one_hot, axes=(1,))
        return softmax_loss_.sum() / logits.shape[0]
        


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(self.dim), device=self.device, dtype=self.dtype)
        self.bias = Parameter(init.zeros(self.dim), device=self.device, dtype=self.dtype)
        self.running_mean = init.zeros(dim, device=self.device, dtype=self.dtype)
        self.running_var = init.ones(dim, device=self.device, dtype=self.dtype)
        

    def forward(self, x: Tensor) -> Tensor:
        
        # find axis
        w = self.weight.broadcast_to(x.shape)
        b = self.bias.broadcast_to(x.shape)
        if self.training:            
            mu = x.sum(axes=(0,), keepdims=True) / x.shape[0]   # (1, n_features)
            var = ( (x - mu.broadcast_to(x.shape)) ** 2).sum(axes=(0,), keepdims=True) / x.shape[0]   # (1, n_features) 
            v = var.broadcast_to(x.shape) # (bs, n_features)
            den = (v + self.eps) ** (0.5) # (bs, n_features)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.reshape(self.running_mean.shape).detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.reshape(self.running_var.shape).detach()
            return w * (x - mu.broadcast_to(x.shape)) / den +  b
        else:
            # in eval mode
            mean = self.running_mean.broadcast_to(x.shape)
            var = self.running_var.broadcast_to(x.shape)
            num = x - mean
            den = (var + self.eps) ** (0.5)            
            return w * (num / den) + b
        


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape # nchw
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))  # nhw,c
        y = super().forward(_x)
        y = y.reshape((s[0], s[2], s[3], s[1])) # n,h,w,c   TO-DO: Bug Occurs!
        y = y.transpose((2,3)).transpose((1,2))
        return y # n,c,h,w


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(self.dim), dtype=self.dtype, device=self.device)    # (self.dim,)
        self.bias = Parameter(init.zeros(self.dim), dtype=self.dtype, device=self.device)     # (self.dim,)
        

    def forward(self, x: Tensor) -> Tensor:
        
        # find axis
        axis = None
        for axis_, dim_ in enumerate(reversed(x.shape)):
            if dim_ == self.dim:
                axis = len(x.shape) - 1 - axis_
                break
        
        mu = x.sum(axes=(axis,), keepdims=True) / x.shape[axis]        
        var = ( (x - mu.broadcast_to(x.shape)) ** 2 ).sum(axes=(axis,)) / x.shape[axis]
        den = (var + self.eps)**(0.5)
        den = den.reshape(mu.shape).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * (x - mu.broadcast_to(x.shape)) / den +  self.bias.broadcast_to(x.shape)
        


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        
        if self.training:
            x = x * init.randb(*x.shape, p=1-self.p, device=x.device, requires_grad=False) / (1 - self.p)
        return x        
        


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        
        return x + self.fn(x)
        

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        
        self.device = device
        self.dtype = dtype
        fan_in = kernel_size * kernel_size * in_channels
        fan_out = kernel_size * kernel_size * out_channels        

        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        weight_param = init.kaiming_uniform(fan_in, fan_out, shape=weight_shape, device=self.device, dtype=self.dtype)
        self.weight = Parameter(weight_param, device=self.device, dtype=dtype)
        self.b = bias
        if bias:
            bias_shape = (out_channels,)
            # bias_param = init.kaiming_uniform(fan_in, fan_out, shape=bias_shape)
            bias_param = init.rand(*bias_shape, low=-1, high=1, device=self.device, dtype=self.dtype) * 1.0 / math.sqrt(kernel_size * kernel_size * in_channels)
            # bias_param = init.rand(*bias_shape, low=-1.0 / math.sqrt(fan_in), high=1.0 / math.sqrt(fan_in), device=self.device, dtype=self.dtype)
            self.bias = Parameter(bias_param, device=self.device, dtype=dtype)
        

    def forward(self, x: Tensor) -> Tensor:
        
        N, C_in, H, W = x.shape
        x = x.transpose(axes=(1,2)).transpose(axes=(2,3))  # nhwc
        # (h - k + 2*p)/s + 1 = h
        # h - k + 2*p = s * (h-1)
        # p = (s*(h-1) - h + k) / 2
        # (2*31 - 32 + 3) // 2 = 33 // 2 = 16
        # padding = (self.stride * (H - 1) - H + self.kernel_size) // 2
        padding = self.kernel_size // 2
        x = ndl.ops.conv(x, self.weight, stride=self.stride, padding=padding)
        if self.b:
            x = x + self.bias.broadcast_to(x.shape) # nhwc
        return x.transpose(axes=(2,3)).transpose(axes=(1,2))
        


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        
        assert nonlinearity in ['tanh', 'relu']
        self.act = None
        if nonlinearity == "tanh":
            self.act = ndl.nn.Tanh()
        elif nonlinearity == "relu":
            self.act = ndl.nn.ReLU()
            
        self.input_size = input_size
        self.hidden_size=  hidden_size
        self.b = bias
        self.device = device
        self.dtype = dtype
        
        # init params
        self.W_ih = Parameter(
            math.sqrt(1.0 / self.hidden_size) * init.rand(self.input_size, self.hidden_size, low=-1., high=1., device=self.device, dtype=self.dtype),
            device = self.device,
            dtype = self.dtype,
            requires_grad = True
        )
        self.W_hh = Parameter(
            math.sqrt(1.0 / self.hidden_size) * init.rand(self.hidden_size, self.hidden_size, low=-1., high=1., device=self.device, dtype=self.dtype),
            device = self.device,
            dtype = self.dtype,
            requires_grad = True
        )
        if self.b:
            self.bias_ih = Parameter(
                math.sqrt(1.0 / self.hidden_size) * init.rand(self.hidden_size, low=-1., high=1., device=self.device, dtype=self.dtype),
                device = self.device,
                dtype = self.dtype,
                requires_grad = True
            )
            self.bias_hh = Parameter(
                math.sqrt(1.0 / self.hidden_size) * init.rand(self.hidden_size, low=-1., high=1., device=self.device, dtype=self.dtype),
                device = self.device,
                dtype = self.dtype,
                requires_grad = True
            )
        

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        
        if h is None:
            h_shape = (X.shape[0], self.hidden_size)
            h = ndl.init.zeros(*h_shape, device=self.device, dtype=self.dtype, requires_grad=True)
        ih = X @ self.W_ih
        hh = h @ self.W_hh

        if self.b:
            ih += self.bias_ih.broadcast_to(ih.shape)
            hh += self.bias_hh.broadcast_to(hh.shape)
        h_ = self.act(
            ih + hh
        )
        return h_
        


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.b = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        
        assert self.nonlinearity in ['relu', 'tanh']
        
        # init parameters
        self.rnn_cells = []
        rnn_cell =  ndl.nn.RNNCell(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            bias = self.b,
            nonlinearity = self.nonlinearity,
            device = self.device,
            dtype = self.dtype
        )
        self.rnn_cells.append(rnn_cell)
        for layer_idx in range(1, self.num_layers):
            rnn_cell =  ndl.nn.RNNCell(
                input_size = self.hidden_size,
                hidden_size = self.hidden_size,
                bias = self.b,
                nonlinearity = self.nonlinearity,
                device = self.device,
                dtype = self.dtype
            )
            self.rnn_cells.append(rnn_cell)
        

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        
        x_split = ndl.ops.split(X, axis=0)
        if h0 is None:
            h0_shape = (self.num_layers, X.shape[1], self.hidden_size)
            h0 = ndl.init.zeros(*h0_shape, device=self.device, dtype=self.dtype, requires_grad=True)
        h0_split = ndl.ops.split(h0, axis=0)
        timestamps = len(x_split)
        h_n = []
        
        for layer_idx, rnn_cell in enumerate(self.rnn_cells):
            
            if layer_idx == 0:
                # initial input
                h = [rnn_cell(x_split[0], h0_split[0])]
                # sequent inputs
                for timestamp in range(1, timestamps):
                    h.append(rnn_cell(x_split[timestamp], h[-1]))  # (timestampes, bs, hidden_size)
            else:
                # multi-layers
                h_ = [rnn_cell(h[0], h0_split[layer_idx])]
                # sequent inputs
                for timestamp in range(1, timestamps):
                    h_.append(rnn_cell(h[timestamp], h_[-1]))
                
                h = h_
                

            h_n.append(h[-1])   # (num_layers, bs, hidden_size)

        output = ndl.ops.stack(h, axis=0)
        h_n = ndl.ops.stack(h_n, axis=0)
        return output, h_n
        


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.b = bias
        self.device = device
        self.dtype = dtype
        
        # init params
        self.W_ih = Parameter(
            math.sqrt(1.0 / self.hidden_size) * init.rand(self.input_size, 4*self.hidden_size, low=-1., high=1., device=self.device, dtype=self.dtype),
            device = self.device,
            dtype = self.dtype,
            requires_grad = True
        )
        self.W_hh = Parameter(
            math.sqrt(1.0 / self.hidden_size) * init.rand(self.hidden_size, 4*self.hidden_size, low=-1., high=1., device=self.device, dtype=self.dtype),
            device = self.device,
            dtype = self.dtype,
            requires_grad = True
        )
        if self.b:
            self.bias_ih = Parameter(
                math.sqrt(1.0 / self.hidden_size) * init.rand(4*self.hidden_size, low=-1., high=1., device=self.device, dtype=self.dtype),
                device = self.device,
                dtype = self.dtype,
                requires_grad = True
            )
            self.bias_hh = Parameter(
                math.sqrt(1.0 / self.hidden_size) * init.rand(4*self.hidden_size, low=-1., high=1., device=self.device, dtype=self.dtype),
                device = self.device,
                dtype = self.dtype,
                requires_grad = True
            )
        
        # define act
        self.sigmoid = ndl.nn.Sigmoid()
        self.tanh = ndl.nn.Tanh()
        


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        
        if h is None:
            h0_shape = (X.shape[0], self.hidden_size)
            c0_shape = (X.shape[0], self.hidden_size)
            h0 = ndl.init.zeros(*h0_shape, device=self.device, dtype=self.dtype, requires_grad=True)
            c0 = ndl.init.zeros(*c0_shape, device=self.device, dtype=self.dtype, requires_grad=True)
        else:
            h0, c0 = h
        
        ih = X @ self.W_ih  # bs, 4*hidden_size
        hh = h0 @ self.W_hh  # bs, 4*hidden_size
        if self.b:
            ih += self.bias_ih.broadcast_to(ih.shape)   # bs, 4*hidden_size
            hh += self.bias_hh.broadcast_to(hh.shape)   # bs, 4*hidden_size
        i, f, g, o = ndl.ops.split(ih + hh, axis=1, sections=self.hidden_size)  # bs, hidden_size
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)
        c_out = f*c0 + i*g              # bs, hidden_size
        h_out = o * self.tanh(c_out)    # bs, hidden_size
        return h_out, c_out
        


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.b = bias
        self.device = device
        self.dtype = dtype
        
        # init parameters
        self.lstm_cells = []
        lstm_cell =  ndl.nn.LSTMCell(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            bias = self.b,
            device = self.device,
            dtype = self.dtype
        )
        self.lstm_cells.append(lstm_cell)
        for layer_idx in range(1, self.num_layers):
            lstm_cell =  ndl.nn.LSTMCell(
                input_size = self.hidden_size,
                hidden_size = self.hidden_size,
                bias = self.b,
                device = self.device,
                dtype = self.dtype
            )
            self.lstm_cells.append(lstm_cell)
        

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        
        x_split = ndl.ops.split(X, axis = 0)
        if h is None:
            h0_shape = (self.num_layers, X.shape[1], self.hidden_size)
            c0_shape = (self.num_layers, X.shape[1], self.hidden_size)
            h0 = ndl.init.zeros(*h0_shape, device=self.device, dtype=self.dtype, requires_grad=True)
            c0 = ndl.init.zeros(*c0_shape, device=self.device, dtype=self.dtype, requires_grad=True)
        else:
            h0, c0 = h
        
        h0_split = ndl.ops.split(h0, axis=0)
        c0_split = ndl.ops.split(c0, axis=0)
        timestamps = len(x_split)
        h_n = []
        c_n = []
        
        for layer_idx, lstm_cell in enumerate(self.lstm_cells):
            
            if layer_idx == 0:
                # initial input
                h, c = lstm_cell(x_split[0], (h0_split[0], c0_split[0]))
                h = [h]
                c = [c]
                # sequent inputs
                for timestamp in range(1, timestamps):
                    h_temp, c_temp = lstm_cell(x_split[timestamp], (h[-1], c[-1]))
                    h.append(h_temp)
                    c.append(c_temp)
            else:
                # multi-layers
                h_temp, c_temp = lstm_cell(h[0], (h0_split[layer_idx], c0_split[layer_idx]))
                # sequent inputs
                h_ = [h_temp]
                c_ = [c_temp]
                for timestamp in range(1, timestamps):
                    h_temp, c_temp = lstm_cell(h[timestamp], (h_[-1], c_[-1]))
                    h_.append(h_temp)
                    c_.append(c_temp)
                
                h = h_
                c = c_
            
            h_n.append(h[-1])
            c_n.append(c[-1])
        
        output = ndl.stack(h, axis=0)
        h_n = ndl.stack(h_n, axis=0)
        c_n = ndl.stack(c_n, axis=0)
        return output, (h_n, c_n)
                
        


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.randn(self.num_embeddings, self.embedding_dim, mean = 0., std = 1.),
            device = self.device,
            dtype = self.dtype,
            requires_grad = True
        )
        

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        
        x_one_hot = ndl.init.one_hot(
            n = self.num_embeddings,
            i = x,
            device = self.device,
            dtype = self.dtype,
            requires_grad = x.requires_grad
        )   # (seq_len, bs, num_embeddings)
        seq_len, bs, _ = x_one_hot.shape
        x_one_hot = x_one_hot.reshape((seq_len*bs, self.num_embeddings))
        
        x = x_one_hot @ self.weight # (seq_len*bs, embedding_dim)
        x = x.reshape((seq_len, bs, self.embedding_dim))
        
        return x
        
