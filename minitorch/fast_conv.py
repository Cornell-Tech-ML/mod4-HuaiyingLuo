from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Index,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Just-in-time compilation decorator with inline optimization."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # TODO: Implement for Task 4.1.
    for out_pos in prange(out_size):
        out_idx: Index = np.zeros(3, np.int32)
        input_idx: Index = np.zeros(3, np.int32)
        weight_idx: Index = np.zeros(3, np.int32)

        to_index(out_pos, out_shape, out_idx)
        batch_idx, out_channel_idx, out_width_idx = out_idx
        conv_sum = 0.0
        for in_channel_idx in range(in_channels):
            for kernel_width_idx in range(kw):
                if not reverse:
                    if (out_width_idx + kernel_width_idx) >= width:
                        conv_sum += 0.0
                    else:
                        weight_idx[0] = out_channel_idx
                        weight_idx[1] = in_channel_idx
                        weight_idx[2] = kernel_width_idx
                        input_idx[0] = batch_idx
                        input_idx[1] = in_channel_idx
                        input_idx[2] = out_width_idx + kernel_width_idx
                        conv_sum += (
                            input[index_to_position(input_idx, s1)]
                            * weight[index_to_position(weight_idx, s2)]
                        )
                else:
                    if (out_width_idx - kernel_width_idx) < 0:
                        conv_sum += 0.0
                    else:
                        weight_idx[0] = out_channel_idx
                        weight_idx[1] = in_channel_idx
                        weight_idx[2] = kernel_width_idx
                        input_idx[0] = batch_idx
                        input_idx[1] = in_channel_idx
                        input_idx[2] = out_width_idx - kernel_width_idx
                        conv_sum += (
                            input[index_to_position(input_idx, s1)]
                            * weight[index_to_position(weight_idx, s2)]
                        )

        out[index_to_position(out_idx, out_strides)] = conv_sum


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass of the 1D Convolution.

        Args:
        ----
            ctx : Context
            grad_output : batch x out_channel x w

        Returns:
        -------
            tuple[Tensor, Tensor]

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides

    # TODO: Implement for Task 4.2.
    for out_pos in prange(out_size):
        out_idx: Index = np.zeros(4, np.int32)
        weight_index: Index = np.zeros(4, np.int32)
        in_idx: Index = np.zeros(4, np.int32)
        to_index(out_pos, out_shape, out_idx)
        batch_idx, out_ch_idx, out_h_idx, out_w_idx = out_idx
        conv_val = 0.0

        for in_ch_idx in range(in_channels):
            for k_h in range(kh):
                for k_w in range(kw):
                    if not reverse:
                        weight_index[0] = out_ch_idx
                        weight_index[1] = in_ch_idx
                        weight_index[2] = k_h
                        weight_index[3] = k_w
                        in_idx[0] = batch_idx
                        in_idx[1] = in_ch_idx
                        in_idx[2] = out_h_idx + k_h
                        in_idx[3] = out_w_idx + k_w
                        if out_h_idx + k_h >= height or out_w_idx + k_w >= width:
                            conv_val += 0.0
                        else:
                            conv_val += (
                                weight[index_to_position(weight_index, s2)]
                                * input[index_to_position(in_idx, s1)]
                            )
                    else:
                        weight_index[0] = out_ch_idx
                        weight_index[1] = in_ch_idx
                        weight_index[2] = k_h
                        weight_index[3] = k_w
                        in_idx[0] = batch_idx
                        in_idx[1] = in_ch_idx
                        in_idx[2] = out_h_idx - k_h
                        in_idx[3] = out_w_idx - k_w
                        if out_h_idx - k_h < 0 or out_w_idx - k_w < 0:
                            conv_val += 0.0
                        else:
                            conv_val += (
                                weight[index_to_position(weight_index, s2)]
                                * input[index_to_position(in_idx, s1)]
                            )
        out[index_to_position(out_idx, out_strides)] = conv_val


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass of the 2D Convolution.

        Args:
        ----
            ctx : Context
            grad_output : batch x out_channel x h x w

        Returns:
        -------
            tuple[Tensor, Tensor]

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
