"""
Module contenant l'architecture du réseau de neurones convolutionnel pour MNIST.
"""

from torch import nn


class ConvNet(nn.Module):
    """
    Réseau de neurones convolutionnel pour la classification d'images MNIST.

    Architecture:
    - 2 couches convolutionnelles avec ReLU et MaxPooling
    - 2 couches fully connected avec ReLU
    - Couche de sortie pour classification
    """

    def __init__(self, input_size=None, n_kernels=6, output_size=10):
        """
        Initialise le réseau ConvNet.

        Args:
            input_size (int, optional): Taille d'entrée (compatibilité, non utilisé)
            n_kernels (int): Nombre de filtres dans les couches convolutionnelles
            output_size (int): Nombre de classes pour la classification
        """
        super().__init__()
        # input_size est gardé pour compatibilité avec l'API existante
        # mais n'est pas utilisé car MNIST a une taille fixe de 28x28
        _ = input_size  # Marquer comme utilisé pour Pylint

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=n_kernels * 4 * 4, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=output_size),
        )

    def forward(self, x):
        """
        Passe avant du réseau.

        Args:
            x (torch.Tensor): Tensor d'entrée de forme (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Logits de sortie de forme (batch_size, output_size)
        """
        return self.net(x)
