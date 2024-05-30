from torch import nn
import torch
import math


class FourierTimeEmbedder(nn.Module):
    def __init__(
        self,
        fourier_dim: int,
        time_embed_dim: int, 
        num_timesteps: int = 0,
        time_scale: float = 1.0, 
        max_period: float = 10000,
    ):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.time_embed_dim = time_embed_dim
        self.time_scale = time_scale    
        self.max_period = max_period
        self.time_embed = nn.Sequential(
            nn.Linear(fourier_dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.num_timesteps = num_timesteps

    def forward(self, t):
        scaled_time = self.time_scale*t/max(self.num_timesteps, 1)
        embedded_time = self.fourier_embedding(scaled_time)
        return self.time_embed(embedded_time)
        
    def fourier_embedding(self, timesteps: torch.Tensor):
        """Create sinusoidal timestep embeddings.

        Args:
            timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
            dim (int): the dimension of the output.
            max_period (int): controls the minimum frequency of the embeddings.
        Returns:
            embedding (torch.Tensor): [N $\times$ dim] Tensor of positional embeddings.
        """
        half = self.fourier_dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        while len(freqs.shape) < len(timesteps.shape) + 1:
            freqs = freqs.unsqueeze(0)

        args = timesteps[..., None].float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.fourier_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding