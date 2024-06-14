from typing import Union

import torch
from torch import nn
import copy
import hydra

from ..networks.time_embedding import FourierTimeEmbedder
from .base import ConditionalNeuralProcess, NeuralProcess


class TimeConditionedNP(nn.Module):
    def __init__(
        self,
        neural_process: Union[ConditionalNeuralProcess, NeuralProcess],
        time_embedder: FourierTimeEmbedder,
        num_timesteps: int = 0, 
        single_model: bool = True,
    ):
        super().__init__()
        # Using one model for all diffusion timesteps (i.e. tying the parameters)
        if single_model:
            self.neural_process = hydra.utils.instantiate(neural_process)
            self.time_embedder = hydra.utils.instantiate(time_embedder)
        
        self.single_model = single_model

        # Using one model per diffusion timestep
        if not single_model:
            self.neural_processes = nn.ModuleList([hydra.utils.instantiate(neural_process)])
            self.time_embedders = nn.ModuleList([hydra.utils.instantiate(time_embedder)])
            for _ in range(num_timesteps):
                self.neural_processes.append(hydra.utils.instantiate(neural_process))
                self.time_embedders.append(hydra.utils.instantiate(time_embedder))

    def forward(
        self, 
        xc: torch.Tensor, 
        yc: torch.Tensor, 
        xt: torch.Tensor, 
        tc: torch.Tensor, 
        tt: torch.Tensor,
    ):
        if not self.single_model:
            # If one model per diffusion timestep, get the model corresponding
            # to current timestep
            time_embedder = self.time_embedders[int(tt[0, 0].item())]
            neural_process = self.neural_processes[int(tt[0, 0].item())]
        else:
            time_embedder = self.time_embedder
            neural_process = self.neural_process
            
        # Get the time embeddings
        # tc will always be 0 since we do not noise the context points up
        time_embedding_tc = time_embedder(tc)
        time_embedding_tt = time_embedder(tt)

        # Concatenate time embedding to context and target embeddings
        xc = torch.cat([xc, time_embedding_tc], dim=-1)
        xt = torch.cat([xt, time_embedding_tt], dim=-1)

        return neural_process.likelihood(
            neural_process.decoder(
                neural_process.encoder(xc, yc, xt), xt), int(tt[0, 0].item())
                )
