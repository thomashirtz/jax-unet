from typing import Sequence, Optional, Type

import jax
import jax.numpy as jnp
from flax import linen as nn


class DoubleConv(nn.Module):
    """
    A module that applies two convolutional layers, each followed by batch normalization and a configurable activation function.

    Args:
        in_channels (int): Number of input channels for the first convolution.
        out_channels (int): Number of output channels for the second convolution.
        mid_channels (Optional[int]): Number of channels for the intermediate convolutional layer, defaults to out_channels.
        activation_function (Type[nn.Module]): Activation function to apply after each convolution (default is ELU).
    """
    in_channels: int
    out_channels: int
    mid_channels: Optional[int] = None
    activation_function: Type[nn.Module] = nn.elu  # ELU by default

    def setup(self):
        """Define layers that make up the DoubleConv block."""
        mid_channels = self.mid_channels if self.mid_channels is not None else self.out_channels
        self.conv1 = nn.Conv(features=mid_channels, kernel_size=(3, 3), padding="SAME", use_bias=False)
        self.bn1 = nn.BatchNorm(use_running_average=True)
        self.conv2 = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME", use_bias=False)
        self.bn2 = nn.BatchNorm(use_running_average=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Perform a forward pass through the two convolutional layers."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation_function(x)
        return x


class Down(nn.Module):
    """
    Downscaling block using max pooling followed by a DoubleConv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply max pooling and then a DoubleConv."""
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return DoubleConv(self.in_channels, self.out_channels)(x)


class Up(nn.Module):
    """
    Upscaling block, optionally using bilinear interpolation or transposed convolution, followed by a DoubleConv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool): Whether to use bilinear interpolation or transposed convolution (default: True).
    """
    in_channels: int
    out_channels: int
    bilinear: bool = True

    @nn.compact
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the upsampling and concatenation followed by a DoubleConv.

        Args:
            x1 (jnp.ndarray): Input from the previous layer.
            x2 (jnp.ndarray): Skip connection from the downsampling path.

        Returns:
            jnp.ndarray: Output after upsampling, concatenation, and DoubleConv.
        """
        if self.bilinear:
            # Bilinear interpolation for upsampling
            x1 = jax.image.resize(x1, shape=(x2.shape[0], x2.shape[1], x2.shape[2], x1.shape[3]), method="bilinear")
        else:
            # Transposed convolution for upsampling
            x1 = nn.ConvTranspose(features=self.in_channels // 2, kernel_size=(2, 2), strides=(2, 2))(x1)

        # Calculate padding to match dimensions
        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]
        x1 = jnp.pad(x1, ((0, 0), (diffY // 2, diffY - diffY // 2), (diffX // 2, diffX - diffX // 2), (0, 0)))

        # Concatenate skip connection
        x = jnp.concatenate([x2, x1], axis=-1)

        # Apply DoubleConv after concatenation
        return DoubleConv(in_channels=self.in_channels, out_channels=self.out_channels)(x)


class OutConv(nn.Module):
    """
    Final convolutional layer that maps to the desired number of output channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (e.g., number of classes).
    """
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply a 1x1 convolution to map to the desired output channels."""
        return nn.Conv(features=self.out_channels, kernel_size=(1, 1))(x)


class UNetwork(nn.Module):
    """
    U-Net architecture for image segmentation with downsampling and upsampling paths.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channel_list (Sequence[int]): Number of channels at each stage of the U-Net.
        bilinear (bool): Whether to use bilinear upsampling (default: False).
    """
    in_channels: int
    out_channels: int
    channel_list: Sequence[int]
    bilinear: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Perform a forward pass through the U-Net.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Convert NCHW (batch_size, channels, height, width) to NHWC (batch_size, height, width, channels)
        x = jnp.transpose(x, (0, 2, 3, 1))

        # Initial double convolution
        x = DoubleConv(self.in_channels, self.channel_list[0])(x)

        # Downsampling path
        down_results = [x]
        for out_channels in self.channel_list[1:]:
            x = Down(in_channels=x.shape[-1], out_channels=out_channels)(x)
            down_results.append(x)

        # Upsampling path (reversing the channel list except the first)
        for i, out_channels in enumerate(reversed(self.channel_list[:-1])):
            x = Up(in_channels=x.shape[-1], out_channels=out_channels, bilinear=self.bilinear)(x,
                                                                                               down_results[-(i + 2)])

        # Final output layer
        x = OutConv(in_channels=x.shape[-1], out_channels=self.out_channels)(x)

        # Convert NHWC back to NCHW before returning
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x
