import torch
from torch import nn
from typing import Dict, Any

from .module import (
    Atomwise,
    Ewald
)

__all__ = ['Les']

class Les(nn.Module):

    def __init__(self, les_arguments: Dict[str, Any]):
        """
        LES model for long-range interations
        """
        super().__init__()

        self._parse_arguments(les_arguments)

        self.atomwise = Atomwise(
        n_layers=self.n_layers,
        n_hidden=self.n_hidden,
        add_linear_nn=self.add_linear_nn
    )

        self.ewald = Ewald(
            sigma=self.sigma,
            dl=self.dl
            )

    def _parse_arguments(self, les_arguments: Dict[str, Any]):
        """
        Parse arguments for LES model
        """
        self.n_layers = les_arguments.get('n_layers', 3)
        self.n_hidden = les_arguments.get('n_hidden', [32, 16])
        self.add_linear_nn = les_arguments.get('add_linear_nn', True)

        self.sigma = les_arguments.get('sigma', 1.0)
        self.dl = les_arguments.get('dl', 2.0)


    def forward(self, 
               desc: torch.Tensor, # [n_atoms, n_features]
               positions: torch.Tensor, # [n_atoms, 3]
               cell: torch.Tensor, # [batch_size, 3, 3]
               batch: torch.Tensor = None,
               ) -> torch.Tensor:
        """
        arguments:
        desc: torch.Tensor
        Descriptors for the atoms. Shape: (n_atoms, n_features)
        positions: torch.Tensor
            positions of the atoms. Shape: (n_atoms, 3)
        cell: torch.Tensor
            cell of the system. Shape: (batch_size, 3, 3)
        batch: torch.Tensor
            batch of the system. Shape: (n_atoms,)
        """
        # check the input shapes
        assert desc.shape[0] == positions.shape[0]
        if batch == None:
            batch = torch.zeros(desc.shape[0], dtype=torch.int64, device=desc.device)

        # compute the latent charges
        latent_charges = self.atomwise(desc, batch)

        # compute the long-range interactions
        E_lr = self.ewald(q=latent_charges,
                          r=positions,
                          cell=cell,
                          batch=batch,
                          )

        output = {
            'E_lr': E_lr,
            'latent_charges': latent_charges,
            'BEC': None,
            }
        return output 
