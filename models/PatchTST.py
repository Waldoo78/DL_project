import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class PatchTSTConfig:
    # Data
    num_channels      : int   = 7
    context_length    : int   = 336
    # Patch
    patch_length      : int   = 16
    patch_stride            : int   = 8
    # Model
    d_model           : int   = 128
    num_heads         : int   = 16
    num_layers        : int   = 3
    ffn_dim           : int   = 256
    dropout           : float = 0.2
    # Norm
    norm_eps          : float = 1e-5
    # Head
    prediction_length : int   = 96
    head_dropout      : float = 0.0



class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    paper: https://openreview.net/pdf?id=cGDAkQo1C0p
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        """
        :param num_channels: Number of channels in the input data
        :param eps: A small value added to the denominator for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_channels))
        self.affine_bias   = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, mode: str):
        if mode == "norm":
            self.get_statistics(x)
            x = self.normalize(x)
        elif mode == "denorm":
            x = self.denormalize(x)
        return x

    def get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.std  = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def normalize(self, x):
        x = (x - self.mean) / self.std
        x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.std + self.mean
        return x
    

class PatchTSTPatchify(nn.Module):
    """
    Patchify the time series sequence into patches.
    Returns:
        torch.Tensor of shape (batch_size, num_channels, num_patches, patch_length)
    """
    def __init__(self, config: PatchTSTConfig):
        """
        :param config: PatchTSTConfig object
        """
        super().__init__()
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride
        self.num_patches  = (config.context_length - config.patch_length) // config.patch_stride + 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, sequence_length, num_channels)
        :return:  (batch_size, num_channels, num_patches, patch_length)
        """
        x = x.transpose(1, 2)                                                      # (batch, channels, L)
        x = F.pad(x, (0, self.patch_stride), mode='replicate')                     # (batch, channels, L+S)
        x = x.unfold(dimension=-1, size=self.patch_length, step=self.patch_stride) # (batch, channels, 42, 16)
        return x

class PatchTSTEmbedding(nn.Module):
    """
    Embed each patch into a d_model dimensional vector.
    Returns:
        torch.Tensor of shape (batch_size, num_channels, num_patches, d_model)
    """
    def __init__(self, config: PatchTSTConfig):
        """
        :param config: PatchTSTConfig object
        """
        super().__init__()
        self.projection= nn.Linear(config.patch_length, config.d_model)
        self.dropout= nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        :param x: (batch_size, num_channels, num_patches, patch_length)
        :return:  (batch_size, num_channels, num_patches, d_model)
        """
        x=self.projection(x)  # (batch, channels, num_patches, d_model)
        x=self.dropout(x)
        return x
        
class PatchTSTPositionalEncoding(nn.Module):
    """
    Leaarnable positional encoding for the patches.
    paper: x_d= Wp * xp +Wpo
    """

    def __init__(self, config: PatchTSTConfig, num_patches: int):
        """
        :param config: PatchTSTConfig object
        """
        super().__init__()
        Wpos=torch.empty(num_patches,config.d_model)
        nn.init.uniform_(Wpos, -0.02, 0.02) # Initialize Wpos with small random values
        self.positional_encoding = nn.Parameter(Wpos, requires_grad=True) #learnable
        self.dropout= nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, num_channels, num_patches, d_model)
        :return:  (batch_size, num_channels, num_patches, d_model)
        """
        x=self.dropout(x + self.positional_encoding)  # (batch, channels, num_patches, d_model)
        return x


class PatchTSTBatchNorm(nn.Module):
    """
    BatchNorm across the channel dimension for each patch.
    paper: x_d= Wp * xp +Wpo
    """

    def __init__(self,config: PatchTSTConfig):
        """
        :param config: PatchTSTConfig object
        """
        super().__init__()
        self.batchnorm=nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size*num_channels, num_patches, d_model)
        :return:  (batch_size*num_channels, num_patches, d_model)
        """
        x=x.transpose(1, 2)  # (batch*channels, d_model, num_patches)
        x=self.batchnorm(x)  
        return x.transpose(1, 2)  # (batch*channels, num_patches, d_model)
    
class PatchTSTEncoderLayer(nn.Module):

    def __init__(self,config:PatchTSTConfig):
        """
        :param config: PatchTSTConfig object
        """
        super().__init__()
        self.self_attn=nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.norm1=PatchTSTBatchNorm(config)
        self.norm2=PatchTSTBatchNorm(config)

        self.ff=nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.dropout=nn.Dropout(config.dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, num_channels, num_patches, d_model)
        :return:  (batch_size, num_channels, num_patches, d_model)
        """
        bs, num_channels, num_patches, d_model = x.shape
        x = x.view(bs * num_channels, num_patches, d_model)  # (batch*channels, num_patches, d_model)

        # Self-Attention + Add & Norm
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN + Add & Norm
        x = self.norm2(x + self.dropout(self.ff(x)))

        return x.view(bs, num_channels, num_patches, d_model)
    

class PatchTSTEncoder(nn.Module):
    """
    Transformer encoder for the patch embeddings.
    Returns:
        torch.Tensor of shape (batch_size, num_channels, num_patches, d_model)
    """
    def __init__(self, config: PatchTSTConfig, num_patches: int):
        """
        :param config: PatchTSTConfig object
        """
        super().__init__()
        self.embedding = PatchTSTEmbedding(config)
        self.positional_encoding = PatchTSTPositionalEncoding(config, num_patches)
        self.layers = nn.ModuleList([PatchTSTEncoderLayer(config) for _ in range(config.num_layers)])  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, num_channels, num_patches, patch_length)
        :return:  (batch_size, num_channels, num_patches, d_model)
        """
        x = self.embedding(x)  # (batch, channels, num_patches, d_model)
        x = self.positional_encoding(x)  # (batch, channels, num_patches, d_model)
        for layer in self.layers:
            x = layer(x)  # (batch, channels, num_patches, d_model)
        return x
 
class PatchTSTHead(nn.Module):
    """
    Final prediction head that maps the encoder output to the desired prediction length.
    Returns:
        torch.Tensor of shape (batch_size, prediction_length, num_channels)
    """
    def __init__(self, config: PatchTSTConfig, num_patches: int):
        """
        :param config: PatchTSTConfig object
        """
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)  # Flatten num_patches and d_model
        self.dropout = nn.Dropout(config.head_dropout)
        self.projection = nn.Linear(num_patches*config.d_model, config.prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, num_channels, num_patches, d_model)
        :return:  (batch_size, num_channels, prediction_length)
        """
        x=self.flatten(x)  # (batch_size, num_channels, num_patches*d_model)
        x=self.dropout(x)
        x=self.projection(x)  # (batch_size, num_channels, prediction_length)
        return x.transpose(1,2) # (batch_size, prediction_length, num_channels)
    

class PatchTST(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.revin    = RevIN(config.num_channels)
        self.patchify = PatchTSTPatchify(config)
        self.encoder  = PatchTSTEncoder(config, num_patches=self.patchify.num_patches)
        self.head     = PatchTSTHead(config, num_patches=self.patchify.num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, sequence_length, num_channels)
        :return:  (batch_size, prediction_length, num_channels)
        """
        x = self.revin(x, mode="norm")    # (batch, sequence_length, num_channels)
        x = self.patchify(x)              # (batch, channels, num_patches, patch_length)
        x = self.encoder(x)               # (batch, channels, num_patches, d_model)
        x = self.head(x)                  # (batch, prediction_length, channels)
        x = self.revin(x, mode="denorm")  # (batch, prediction_length, channels)
        return x
