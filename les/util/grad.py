from typing import Dict
import torch
from torch import nn

def grad(x: torch.Tensor, y: torch.Tensor, training: bool = True) -> torch.Tensor:
    """
    a wrapper for the gradient calculation
    alow multiple dimensional and/or complex y
    y: [n_graphs, ] or [n_graphs, dim_y]
    x: [n_nodes, :]
    """
    if y.is_complex():
        get_imag = True
    else:
        get_imag = False

    if len(y.shape) == 1:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
        gradient_real = torch.autograd.grad(
            outputs=[y],  # [n_graphs, ]
            inputs=[x],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=(training or get_imag),  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]  # [n_nodes, 3]
        if get_imag:
            gradient_imag = torch.autograd.grad(
                outputs=[y/1j],  # [n_graphs, ]
                inputs=[x],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=training,  # Make sure the graph is not destroyed during training
                create_graph=training,  # Create graph for second derivative
                allow_unused=True,  # For complete dissociation turn to true
            )[0]  # [n_nodes, 3]
        else:
            gradient_imag = 0.0
    else:
        dim_y = y.shape[1] 
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y[:,0])]
        gradient_real = torch.stack([
            torch.autograd.grad(
            outputs=[y[:,i]],  # [n_graphs, ]
            inputs=[x],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=(training or (i < dim_y - 1) or get_imag),  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
            )[0] for i in range(dim_y)
           ], axis=2)  # [n_nodes, 3, dim_y]
        # if y is complex, we need to calculate the imaginary part
        if get_imag:
            gradient_imag = torch.stack([
                torch.autograd.grad(
                outputs=[y[:,i]/1j],  # [n_graphs, ]
                inputs=[x],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=(training or (i < dim_y - 1)),  # Make sure the graph is not destroyed during training
                create_graph=training,  # Create graph for second derivative
                allow_unused=True,  # For complete dissociation turn to true
                )[0] for i in range(dim_y)
               ], axis=2)  # [n_nodes, 3, dim_y]
    if get_imag:
        return gradient_real + 1j * gradient_imag
    else:
        return gradient_real
