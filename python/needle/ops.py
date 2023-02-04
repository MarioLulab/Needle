"""Operatpr table."""
# Global operator table.
import operator
from functools import reduce

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy
import math
import sys

from .backend.backend_selection import array_api, NDArray


def prod(x):
    return reduce(operator.mul, x, 1)


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        
        # TO-DO : verify dtype
        return a ** self.scalar
        

    def gradient(self, out_grad, node):
        
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        
        # TO-DO : verify the results
        return a / b
        

    def gradient(self, out_grad, node):
        
        num, den = node.inputs
        lhs = out_grad / den
        rhs = - out_grad * num / power_scalar(den, 2)
        return lhs, rhs
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        
        # TO-DO : verify dtype
        return a / self.scalar
        

    def gradient(self, out_grad, node):
        
        return (out_grad / self.scalar,)
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        
        if array_api is numpy:
            return array_api.swapaxes(a, *(self.axes)) if self.axes is not None else array_api.swapaxes(a, axis1=-2, axis2=-1)
        elif isinstance(a, NDArray):
            inp_axes = list(range(len(a.shape)))
            swap_axis_a = self.axes[0] if self.axes is not None else -1
            swap_axis_b = self.axes[1] if self.axes is not None else -2
            # swap axes
            inp_axes[swap_axis_a], inp_axes[swap_axis_b] = inp_axes[swap_axis_b], inp_axes[swap_axis_a]
            return a.permute(inp_axes)
        else:
            # noted that : torch.transpose not equivalent to numpy.transpose
            raise NotImplementedError
        

    def gradient(self, out_grad, node):
        
        inp = node.inputs[0]
        return out_grad.transpose(self.axes)
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        
        
        return array_api.reshape(a, self.shape)
        

    def gradient(self, out_grad, node):
        
        inp = node.inputs[0]
        return out_grad.reshape(inp.shape)
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        
        inp = node.inputs[0]

        if inp.shape == ():
            broad_axes = tuple([ idx for idx in range(len(out_grad.shape)) ])
            return summation(out_grad, axes=broad_axes)
        else:
            inp_shape_broadcast = [1] * (len(out_grad.shape) - len(inp.shape)) + list(inp.shape)
            broad_axes = []
            for axis, (i,j) in enumerate(zip(inp_shape_broadcast, out_grad.shape)):
                if i != j or (axis < (len(out_grad.shape) - len(inp.shape)) and j == 1):
                    broad_axes.append(axis)
            
            return summation(out_grad, axes=tuple(broad_axes)).reshape(inp.shape)
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims: bool = False):
        # assert axes is None or isinstance(axes, (list, tuple, int))
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        
        return a.sum(self.axes, keepdims=self.keepdims)
        

    def gradient(self, out_grad, node):
        
        inp = node.inputs[0]
        if inp.shape == ():
            inp_shape = (1,)
        else:
            inp_shape = inp.shape
        
        if self.axes is None:
            self.axes = tuple(range(len(inp.shape)))

        out_grad_reshape = [ dim if idx not in self.axes else 1 for idx, dim in enumerate(inp_shape)]
        ret = broadcast_to(reshape(out_grad, out_grad_reshape), inp_shape)
        return ret
        


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims=keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        
        # TO-DO : verify dtype
        dtype_ = 'float32' if 'float32' in [a.dtype, b.dtype] else None
        return a @ b
        

    def gradient(self, out_grad, node):
        
        # out_grad : (m,k) or (k,t,m,k)
        lhs, rhs = node.inputs  # (m,n), (n,k), or (k,t,m,n), (k,t,n,k)
        lhs_grad, rhs_grad = matmul(out_grad, transpose(rhs)), matmul(transpose(lhs), out_grad)

        return lhs_grad, rhs_grad
        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        
        return -a
        

    def gradient(self, out_grad, node):
        
        inp = node.inputs[0]
        return -1 * out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        
        # TO-DO : verify dtype
        return array_api.log(a)
        

    def gradient(self, out_grad, node):
        
        input_node = node.inputs[0]
        return out_grad / input_node
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        
        # TO-DO : verify dtype
        return array_api.exp(a)
        

    def gradient(self, out_grad, node):
        
        input_node = node.inputs[0]
        return out_grad * exp(input_node)
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a : NDArray):
        
        return array_api.maximum(a, 0)
        

    def gradient(self, out_grad, node):
        
        ret = out_grad * (node.realize_cached_data() > 0.)
        return ret
        


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        
        # TO-DO : verify the results
        Z_max = Z.max(axis=self.axes, keepdims=True)    # (5,1)
        Z_max_broadcast = array_api.broadcast_to(Z_max, Z.shape)    # e.g. : (5,1) -> (5,3)
        Z_exp = array_api.exp(Z - Z_max_broadcast)
        Z_summation = array_api.summation(Z_exp, axis=self.axes)
        
        return array_api.log(Z_summation) + Z_max.reshape(Z_summation.shape)
        

    def gradient(self, out_grad, node):
        
        input_node = node.inputs[0]
        Z_max = input_node.realize_cached_data().max(axis=self.axes, keepdims=True)
        Z_max = Tensor(Z_max, device=input_node.device, dtype=input_node.dtype, requires_grad=False)
        Z_max = Z_max.broadcast_to(input_node.shape)
        nom = exp(input_node - Z_max)
        den = summation(exp(input_node - Z_max), axes=self.axes, keepdims=True)
        den = den.broadcast_to(input_node.shape)
        softmax_val = nom / den

        unsqueeze_shape = list(input_node.shape)
        axes = range(len(input_node.shape)) if self.axes is None else self.axes
        for axis in axes:
            unsqueeze_shape[axis] = 1
        
        return out_grad.reshape(unsqueeze_shape).broadcast_to(softmax_val.shape) * softmax_val
        

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        
        return array_api.tanh(a)
        

    def gradient(self, out_grad, node):
        
        input_node = node.inputs[0]
        return out_grad * (1 - power_scalar(tanh(input_node), 2)) 
        


def tanh(a):
    return Tanh()(a)

class Sigmoid(TensorOp):
    def compute(self, a):
        
        return (1 + array_api.exp(-a)) ** (-1)
        

    def gradient(self, out_grad, node):
        
        input_node = node.inputs[0]
        return out_grad * sigmoid(input_node) * (1 - sigmoid(input_node))
        


def sigmoid(a):
    return Sigmoid()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args) -> Tensor:
        
        device_ = args[0].device
        dtype_ = args[0].dtype
        # requires_grad_ = args[0].requires_grad
        # 分别将 args 要变的axis挪到最后一维, compact(), 然后flat,赋值给stacked_array
        permute_axes = tuple([a for a in range(args[0].ndim) if a != self.axis]) + (self.axis,)
        
        
        shape_ = (len(args), prod(args[0].shape))
        stacked_array = array_api.empty(shape=shape_, device=device_, dtype=dtype_)
        for i in range(len(args)):
            stacked_array[i,:] = args[i].permute(permute_axes).compact().flat
            
        reshape_shape = (len(args),) + tuple([args[0].shape[a] for a in permute_axes])
        stacked_array = stacked_array.reshape(reshape_shape)
        
        stacked_array_axes = [i for i in range(stacked_array.ndim)]
    
        head_axis = stacked_array_axes.pop(0)
        tail_axis = stacked_array_axes.pop(-1)
        stacked_array_axes.insert(self.axis, tail_axis)
        stacked_array_axes.insert(self.axis, head_axis)
        return stacked_array.permute(stacked_array_axes).compact()
        


    def gradient(self, out_grad, node):
        
        return split(out_grad, axis=self.axis)
        


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int, sections: int = None):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        sections - sections. if None, split result will reduce dimension `axis`, else will keep dimension `axis`.
        """
        self.axis = axis
        self.sections = sections

    def compute(self, A):
        
        permute_axes = (self.axis,) + tuple([a for a in range(A.ndim) if a != self.axis])
        A_permuted = A.permute(permute_axes)
        if self.sections is None:
            # reduce dimension `axis`
            A_split_list = [A_permuted[i].compact().reshape(A_permuted.shape[1:]) for i in range(A_permuted.shape[0])]
        else:
            # keep dimension `axis`
            assert A_permuted.shape[0] % self.sections == 0, "Only support shape[0] is an integral multiple of sections Now"
            A_split_list = [A_permuted[i : i+self.sections].compact().reshape((self.sections, *A_permuted.shape[1:])).permute(permute_axes) for i in range(0, A_permuted.shape[0], self.sections)]

        return A_split_list
        

    def gradient(self, out_grad, node):
        
        return stack(out_grad, axis=self.axis).reshape(node.inputs[0].shape)
        

def split(a, axis, sections: int = None):
    return Split(axis, sections=sections)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        
        return a.flip(self.axes)
        

    def gradient(self, out_grad, node):
        
        return flip(out_grad, axes=self.axes)
        


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        
        orig_shape = a.shape
        dilate_shape = list(a.shape)
        for i, s in enumerate(a.shape):
            if i in self.axes:
                dilate_shape[i] = orig_shape[i] * (1 + self.dilation)
        
        dilate_a = array_api.full(dilate_shape, 0., device=a.device)
        # create slices
        indices = tuple()
        for i, s in enumerate(dilate_a.shape):
            if i in self.axes:
                indices += (slice(0, s, 1+self.dilation),)
            else:
                indices += (slice(0, s, 1),)
        
        dilate_a[indices] = a
        return dilate_a
        

    def gradient(self, out_grad, node):
        
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)
        


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        
        dilate_shape = a.shape
        undilate_shape = list(a.shape)
        for i, s in enumerate(a.shape):
            if i in self.axes:
                undilate_shape[i] = dilate_shape[i] // (1 + self.dilation)

        indices = tuple()
        for i, s in enumerate(dilate_shape):
            if i in self.axes:
                indices += (slice(0, s, 1+self.dilation),)
            else:
                indices += (slice(0, s, 1),)
        return a[indices]
        

    def gradient(self, out_grad, node):
        
        return dilate(out_grad, axes=self.axes, dilation=self.dilation)
        


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # A : X with shape (N, H, W, C_in)
        # B : weights with shape (K, K, C_in, C_out)
        
        # assert self.stride == 1, "Only Support `stride == 1` Now"
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        
        padding_axes = ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0))
        A_padding = A.pad(axes=padding_axes)
        Ns, Hs, Ws, Cs = A_padding.strides
        
        inner_dim = K * K * C_in
        im2col_shape = (N, (H-K+2*self.padding) // self.stride + 1, (W-K+2*self.padding) // self.stride + 1, K, K, C_in)
        A_padding_as_strided = A_padding.as_strided(
            shape = im2col_shape,
            strides = (Ns, self.stride * Hs, self.stride * Ws, Hs, Ws, Cs)
        ).compact()
        A_padding_as_strided = A_padding_as_strided.reshape((prod(im2col_shape[:3]), inner_dim))
        
        out = A_padding_as_strided @ B.compact().reshape((inner_dim,C_out))
        out_reshape = im2col_shape[:3] + (C_out,)
        return out.reshape(out_reshape)
        

    def gradient(self, out_grad, node):
        
        # X       : (N, H, W, C_in)
        # Weights : (K, K, C_in, C_out)
        # out_grad : (N, (H-K+2*self.padding)//self.stride + 1, (W-K+2*self.padding)//self.stride + 1, C_out)
        
        # assert self.padding == 0
        # assert self.stride == 1
        
        X, weights = node.inputs[0], node.inputs[1]
        N,H,W,C_in = X.shape
        K,_,_,C_out = weights.shape
        
        out_grad_dalited = dilate(out_grad, axes=(1,2), dilation=(self.stride-1))
        weights_flipped = flip(weights, axes=(0,1))
        weights_transpose = transpose(weights_flipped)  # (K,K,C_out,C_in)
                
        # padding_for_conv_X = K - 1 - self.padding
        # padding_for_conv_X = (self.stride - 1) * H // 2 + K - self.padding - self.stride
        padding_for_conv_X = math.ceil((2*K - 2*self.padding - self.stride - 1) / 2)
        
        kernel_size_for_conv_X = K
        
        X_grad = conv(out_grad_dalited, weights_transpose, stride=1, padding=padding_for_conv_X)    # expected to be (N, H, W, C_in)


        X_permuted = X.transpose(axes=(0,3))  # (C_in, H, W, N)
        
        # out_grad_permuted = out_grad.permute((1,2,0,3))  # (H-K+2*self.padding)//self.stride + 1, W-K+2*self.padding)//self.stride + 1), N, C_out)
        out_grad_permuted = out_grad.transpose(axes=(0,1)).transpose(axes=(1,2))
        out_grad_permuted_dilated = dilate(out_grad_permuted, axes=(0,1), dilation=(self.stride-1))
        weights_grad_conved = conv(X_permuted, out_grad_permuted_dilated, stride=1, padding=self.padding)   # (C_in, K, K, C_out)    
        # weights_grad = weights_grad_conved.permute((1,2,0,3)) # expected to be (K, K, C_in, C_out)
        weights_grad = weights_grad_conved.transpose(axes=(0,1)).transpose(axes=(1,2)) # expected to be (K, K, C_in, C_out)
        return X_grad, weights_grad
        


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



