from typing import Sequence, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from unet.utilities import check_dtype_support


DTYPE : jnp.dtype | None = jnp.float32  # Set to jnp.bfloat16 for lower precision
check_dtype_support(dtype=DTYPE)


class DoubleConv(nn.Module):
    """
    A module that applies two convolutional layers, each followed by a configurable activation function.

    Args:
        in_channels (int): Number of input channels for the first convolution.
        out_channels (int): Number of output channels for the second convolution.
        activation (Callable[[jnp.ndarray], jnp.ndarray]): Activation function to apply after each convolution (default is ReLU).
    """
    in_channels: int
    out_channels: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu  # ReLU by default

    def setup(self):
        """Define the two convolutional layers."""
        self.conv1 = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=True,
            dtype=DTYPE,
            data_format='NHWC'
        )
        self.conv2 = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding='SAME',
            use_bias=True,
            dtype=DTYPE,
            data_format='NHWC'
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply two convolutions with activation."""
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
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

    def setup(self):
        """Define the pooling layer and the DoubleConv."""
        self.pool = nn.max_pool(
            window_shape=(2, 2),
            strides=(2, 2),
            padding='SAME',
            data_format='NHWC'
        )
        self.double_conv = DoubleConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            activation=nn.relu  # You can customize this if needed
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply max pooling followed by DoubleConv."""
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    """
    Upscaling block using either bilinear interpolation or transposed convolution, followed by a DoubleConv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool): Whether to use bilinear upsampling (default: True).
    """
    in_channels: int
    out_channels: int
    bilinear: bool = True

    def setup(self):
        """Define the upsampling method and the DoubleConv."""
        if self.bilinear:
            self.up = nn.UpSampling(scale=(2, 2), method='bilinear')
        else:
            self.up = nn.ConvTranspose(
                features=self.out_channels,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding='SAME',
                dtype=DTYPE,
                data_format='NHWC'
            )
        self.double_conv = DoubleConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            activation=nn.relu  # You can customize this if needed
        )

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Perform upsampling, concatenate with the skip connection, and apply DoubleConv.

        Args:
            x1 (jnp.ndarray): Input from the previous layer.
            x2 (jnp.ndarray): Skip connection from the downsampling path.

        Returns:
            jnp.ndarray: Output after upsampling, concatenation, and DoubleConv.
        """
        x1 = self.up(x1)

        # Ensure the upsampled tensor has the same spatial dimensions as the skip connection
        x1_shape = x1.shape
        x2_shape = x2.shape
        if x1_shape[1] != x2_shape[1] or x1_shape[2] != x2_shape[2]:
            x1 = jax.image.resize(
                x1,
                shape=(x2_shape[0], x2_shape[1], x2_shape[2], x1_shape[3]),
                method='bilinear'
            )

        # Concatenate along the channels axis
        x = jnp.concatenate([x2, x1], axis=-1)

        # Apply DoubleConv
        x = self.double_conv(x)
        return x


class OutConv(nn.Module):
    """
    Final convolutional layer that maps to the desired number of output channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (e.g., number of classes).
    """
    in_channels: int
    out_channels: int

    def setup(self):
        """Define the final 1x1 convolutional layer."""
        self.conv = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            padding='SAME',
            use_bias=True,
            dtype=DTYPE,
            data_format='NHWC'
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the 1x1 convolution."""
        x = self.conv(x)
        return x


class UNetwork(nn.Module):
    """
    U-Net architecture for image segmentation with downsampling and upsampling paths.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channel_list (Sequence[int]): Number of channels at each stage of the U-Net.
        bilinear (bool): Whether to use bilinear upsampling (default: True).
    """
    in_channels: int
    out_channels: int
    channel_list: Sequence[int]
    bilinear: bool = True

    def setup(self):
        """Define the U-Net architecture by setting up encoder and decoder paths."""
        # Encoder path
        self.downs = [
            DoubleConv(
                in_channels=self.in_channels if i == 0 else self.channel_list[i - 1],
                out_channels=feat,
                activation=nn.relu  # You can customize this if needed
            )
            for i, feat in enumerate(self.channel_list)
        ]
        self.pools = [
            nn.max_pool(
                window_shape=(2, 2),
                strides=(2, 2),
                padding='SAME',
            )
            for _ in self.channel_list
        ]

        # Decoder path
        self.ups = [
            Up(
                in_channels=feat * 2,  # Because of concatenation
                out_channels=self.channel_list[-i - 2],
                bilinear=self.bilinear
            )
            for i, feat in enumerate(self.channel_list[::-1][1:])
        ]

        # Final output layer
        self.out_conv = OutConv(
            in_channels=self.channel_list[0],
            out_channels=self.out_channels
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Perform a forward pass through the U-Net.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, height, width, in_channels).

        Returns:
            jnp.ndarray: Output tensor of shape (batch_size, height, width, out_channels).
        """
        encoder_features = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            encoder_features.append(x)
            x = self.pools[self.downs.index(down)](x)

        # Decoder path
        for up in self.ups:
            skip_connection = encoder_features.pop()
            x = up(x, skip_connection)

        # Final output layer
        x = self.out_conv(x)
        return x
