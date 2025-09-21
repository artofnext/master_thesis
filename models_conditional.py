import torch
import torch.nn as nn
import math
from einops import rearrange  #pip install einops; conda install esri::einops
import torch.nn.functional as F
from typing import List


class PeriodicalEmbeddings(nn.Module):
    def __init__(self, num_steps: int, emb_dim: int):
        super().__init__()
        pos = torch.arange(num_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        emb = torch.zeros(num_steps, emb_dim, requires_grad=False)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div)
        self.emb = emb

    def forward(self, x, t):
        embeds = self.emb[t].to(x.device)
        return embeds[:, :, None, None]


class ConditionalEmbeddings(nn.Module):
    def __init__(self, input_channels:int, num_channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # self.down = nn.AdaptiveAvgPool2d((emb_dim, emb_dim))

    def forward(self, y):  # y is a low-res image for conditional embedding
        y = F.interpolate(y, scale_factor=2, mode='nearest')  # upscale LR to match the SR
        c_emb = self.relu(self.conv1(y))
        c_emb = self.relu(self.conv2(c_emb))
        # c_emb = self.down(c_emb)
        return c_emb


class ImageClassEmbedding(nn.Module):
    def __init__(self, num_classes: int, layer_channels: List[int], base_emb_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, base_emb_dim)  # Common learnable embedding for classes
        self.layer_adapters = nn.ModuleList(
            [nn.Linear(base_emb_dim, ch_dim) for ch_dim in layer_channels]
        )  # One adapter per U-Net layer


    def forward(self, x, class_ids):
        """
        Converts class IDs (e.g., [0, 1, 2]) into embeddings.

        :param x:
        :param class_ids: Tensor of shape (batch_size,)
        :return: List of embeddings, one per layer. Each embedding matches a layer's channel dimensionality.
        """
        common_emb = self.embedding(class_ids).to(x.device)
        adapted_emb = [adapter(common_emb).to(x.device) for adapter in self.layer_adapters]
        return adapted_emb


class FiLM(nn.Module):
    def __init__(self, embedding_dim, num_channels):
        super().__init__()
        self.scale = nn.Linear(embedding_dim, num_channels)  # To modulate normalization scale
        self.shift = nn.Linear(embedding_dim, num_channels)  # To modulate normalization shift

    def forward(self, x, class_emb):
        """
        Applies scaling and shifting to input `x` based on `class_emb`.

        :param x: Input tensor of shape (batch_size, num_channels, height, width)
        :param class_emb: Class embedding of shape (batch_size, embedding_dim)
        :return: Modulated tensor `x`.
        """

        gamma = self.scale(class_emb)[:, :, None, None].to(x.device)  # Scale factors
        beta = self.shift(class_emb)[:, :, None, None].to(x.device)  # Shift factors

        return gamma * x + beta  # Feature-wise Linear Modulation


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, dropout: float):
        super().__init__()

        self.num_channels = num_channels

        self.relu = nn.ReLU(inplace=True)
        self.gr_norm_1 = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.gr_norm_2 = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        self.film1 = FiLM(embedding_dim=num_channels, num_channels=num_channels)
        self.film2 = FiLM(embedding_dim=num_channels, num_channels=num_channels)


        # Add BatchNorm layer for each conv
        # self.batch_norm1 = nn.BatchNorm2d(num_channels)
        # self.batch_norm2 = nn.BatchNorm2d(num_channels)



    def forward(self, x, t_emb, cond_emb, class_emb):
        x = x + t_emb[:, :x.shape[1], :, :]  # concatenate time embedding

        # print("x shape ", x.shape)
        # print("number of channels ", self.num_channels)
        # print("cond_emb shape ", cond_emb.shape)

        # add conditional embedding
        cond_emb = cond_emb.repeat(1, x.shape[1], 1, 1)
        cond_emb = F.interpolate(cond_emb, size=x.shape[2:], mode='nearest')
        x = x + cond_emb

        r = self.relu(self.film1(self.gr_norm_1(x), class_emb))  # 1-st group normalization and conditional class modulation and activation
        r = self.conv1(r)
        r = self.dropout(r)
        r = self.relu(self.film2(self.gr_norm_2(r), class_emb))  # 2-nd group normalization and conditional class modulation and activation
        r = self.conv2(r)

        # First forward pass with BatchNorm
        # r = self.conv1(self.relu(self.gr_norm_1(x)))
        # r = self.batch_norm1(r)  # Apply BatchNorm after Conv1
        # r = self.dropout(r)

        # Second forward pass with BatchNorm
        # r = self.conv2(self.relu(self.gr_norm_2(r)))
        # r = self.batch_norm2(r)  # Apply BatchNorm after Conv2

        return r + x


# class Attn(nn.MultiheadAttention):
#     def __init__(self, embed_dim, num_heads, dropout=0.0):
#         super().__init__(self, embed_dim, num_heads, dropout)

# TODO rename to Attention
class Attention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float, biased: bool = True):  # TODO change to the True by default
        super().__init__()
        # ensure num_channels is divisible by num_heads
        assert num_channels % num_heads == 0, f"Number of channels ({num_channels}) must be divisible by number of heads ({num_heads})"
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.head_dim = num_channels // num_heads
        # linear projection for Q, K, and V, output size is 3 * num_channels to get the three tensors
        self.qkv_proj = nn.Linear(num_channels, num_channels * 3, bias=biased)

        # output linear projection
        self.out_proj = nn.Linear(num_channels, num_channels, bias=biased)
        self.dropout_probability = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # reshape the input from 'b c h w' to 'b (h w) c' and then apply the linear projection.
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        qkv = self.qkv_proj(x_flat)

        # split the qkv tensor into q, k, and v and reshape for multi-head attention.
        # 'qkv' has shape (b, h*w, 3*c). 'rearrange' is used to split it into 'q', 'k', and 'v' of shape (b, num_heads, h*w, head_dim).
        q, k, v = rearrange(
            qkv, 'b L (K H C_head) -> K b H L C_head', K=3, H=self.num_heads
        )

        # compute the scaled dot-product attention.
        # F.scaled_dot_product_attention is optimized version.
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout_probability
        )

        # reshape the attention output back and apply the final projection, combine the heads back into a single tensor of shape (b, h*w, c).
        attn_output = rearrange(
            attn_output, 'b H L C_head -> b L (H C_head)'
        )

        # final projection.
        output = self.out_proj(attn_output)

        # reshape the output back to the original image format
        return rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)

# # TODO delete
# class AttentionAlt(nn.Module):
#     def __init__(self, num_channels: int, num_heads: int, dropout: float):
#         super().__init__()
#         # ensure num_channels is divisible by num_heads
#         assert num_channels % num_heads == 0, f"Number of channels ({num_channels}) must be divisible by number of heads ({num_heads})"
#         self.qkv_proj = nn.Linear(num_channels, num_channels * 3)
#         self.out_proj = nn.Linear(num_channels, num_channels)
#         self.num_heads = num_heads
#         self.dropout_probability = dropout
#
#     def forward(self, x):
#         h, w = x.shape[2:]
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x = self.qkv_proj(x)
#         x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
#         q, k, v = x[0], x[1], x[2]
#         x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout_probability)
#         x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
#         x = self.out_proj(x)
#         return rearrange(x, 'b h w C -> b C h w')
#
# # TODO delete
# class AttentionOld(nn.Module):
#     def __init__(self, num_channels: int, num_heads: int, dropout: float):
#         super().__init__()
#         self.proj1 = nn.Linear(num_channels, num_channels * 3)
#         self.proj2 = nn.Linear(num_channels, num_channels)
#         self.num_heads = num_heads
#         self.dropout_prob = dropout
#
#     def forward(self, x):
#         h, w = x.shape[2:]
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x = self.proj1(x)
#         x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
#         q, k, v = x[0], x[1], x[2]
#         x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout_prob)
#         x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
#         x = self.proj2(x)
#         return rearrange(x, 'b h w C -> b C h w')


class ULayer(nn.Module):
    def __init__(self,
                 upscale: bool,
                 attention: bool,
                 num_groups: int,
                 dropout: float,
                 num_heads: int,
                 ch: int):
        super().__init__()
        self.ResidualBlock1 = ResidualBlock(num_channels=ch, num_groups=num_groups, dropout=dropout)
        self.ResidualBlock2 = ResidualBlock(num_channels=ch, num_groups=num_groups, dropout=dropout)
        if upscale:
            # self.conv = nn.ConvTranspose2d(ch, ch // 2, kernel_size=3, stride=2, padding=1)
            self.conv = nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attn_layer = Attention(ch, num_heads=num_heads, dropout=dropout)

    def forward(self, x, t_emb, cond_emb, class_emb):
        x = self.ResidualBlock1(x, t_emb, cond_emb, class_emb)
        if hasattr(self, 'attn_layer'):
            x = self.attn_layer(x)
        x = self.ResidualBlock2(x, t_emb, cond_emb, class_emb)
        return self.conv(x), x


class UNET(nn.Module):
    def __init__(self,
                 chnls=None,  # list of integers that represents channels
                 attns=None,  # List of booleans for attention layer
                 num_classes : int = 4,  # class guided diffusion
                 num_groups: int = 32,
                 dropout: float = 0.1,
                 num_heads: int = 8,
                 input_channels: int = 1,
                 output_channels: int = 1,
                 time_steps: int = 1000):
        super().__init__()
        if chnls is None:
            chnls = [64, 128, 256, 512, 512, 384]  # default value if parameter is none
        if attns is None:
            attns = [False, False, True, False, False, True]  # default value if parameter is none
        assert len(chnls) == len(attns)
        self.num_layers = len(chnls)
        self.shallow_conv = nn.Conv2d(input_channels, chnls[0], kernel_size=3, padding=1)
        preout_channels = (chnls[-1] // 2) + chnls[0]
        self.late_conv = nn.Conv2d(preout_channels, preout_channels // 2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(preout_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = PeriodicalEmbeddings(num_steps=time_steps, emb_dim=max(chnls))

        self.class_embedding = ImageClassEmbedding(num_classes=num_classes,
                                                   layer_channels=chnls,
                                                   base_emb_dim=max(chnls)  # TODO maybe too much?
                                                   )

        # self.conditional = ConditionalEmbeddings(input_channels=input_channels, num_channels=64)
        for i in range(self.num_layers):
            # create unet layers
            layer = ULayer(
                upscale=self.is_upscale(i),
                attention=attns[i],
                num_groups=num_groups,
                dropout=dropout,
                ch=chnls[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i + 1}', layer)
            # conditional_emb = ConditionalEmbeddings(input_channels=input_channels, num_channels=Channels[i])
            # setattr(self, f'Conditional{i + 1}', conditional_emb)


    def is_upscale(self, layer_num: int):
        """
        Determines if a given layer is considered an upscale layer based on the
        layer index and the total number of layers.

        :param layer_num: Index of the layer in the sequence
        :type layer_num: int
        :return: True if the layer is in the second half of the sequence,
                 indicating an upscale layer; False otherwise
        :rtype: bool
        """
        #  for first half layers returns False, then True
        return layer_num >= self.num_layers//2


    def forward(self, x, t, y, ci):
        x = self.shallow_conv(x)
        residuals = []
        t_emb = self.embeddings(x, t)
        # cond_emb = self.conditional(y)
        cond_emb = y  # conditional embedding shortcut
        class_embs = self.class_embedding(x, ci)
        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i + 1}')
            x, r = layer(x, t_emb, cond_emb, class_embs[i])
            residuals.append(r)
        for j in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{j + 1}')

            x = layer(x, t_emb, cond_emb, class_embs[j])[0]  # take only first element, the second is residual and not needed
            resds = residuals[self.num_layers - j - 1]
            x = torch.concat((x, resds), dim=1)

        x = self.output_conv(self.relu(self.late_conv(x)))

        return x


class Scheduler(nn.Module):
    """
    Manages the noise schedule for a diffusion model.

    This scheduler pre-computes the `beta` and `alpha` values for each time step.
    `beta` controls the amount of noise added at each step, while `alpha` is a
    cumulative product of `(1 - beta)` values, used to calculate the noise level
    at any given time step in a closed-form solution.

    Args:
        num_time_steps (int, optional): The total number of diffusion time steps.
                                        Defaults to 500.
    """
    def __init__(self, num_time_steps: int = 500):
        super().__init__()

        # pre-compute a linear schedule for beta.
        # beta values are small positive numbers that increase linearly over time steps, `requires_grad=False` make it non-trainable.
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        # Calculate alpha.
        alpha = 1 - self.beta

        # calculate the cumulative product of alpha. This is allowing for direct calculation of the noise level at
        # any time step without iterating through all previous steps, `requires_grad_(False)` make it non-trainable.
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        """
        Returns the beta and alpha values for a specific time step.

        This method acts as a lookup for the pre-computed values, providing
        the necessary parameters for the diffusion process at time `t`.

        Args:
            t (torch.Tensor): A tensor representing the time steps.

        Returns:
            tuple: A tuple containing the `beta` and `alpha` values at time `t`.
        """
        return self.beta[t], self.alpha[t]



