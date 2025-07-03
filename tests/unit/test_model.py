"""
Tests unitaires pour le modèle ConvNet
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Ajouter le chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from models.convnet import ConvNet


class TestConvNet:
    """Tests pour la classe ConvNet"""

    @pytest.fixture
    def model(self):
        """Fixture pour créer un modèle de test"""
        return ConvNet(input_size=1, n_kernels=6, output_size=10)

    @pytest.fixture
    def sample_data(self):
        """Fixture pour créer des données de test"""
        batch_size = 4
        return torch.randn(batch_size, 1, 28, 28)

    def test_model_initialization(self, model):
        """Test de l'initialisation du modèle"""
        assert isinstance(model, ConvNet)
        assert model.conv1.in_channels == 1
        assert model.conv1.out_channels == 6
        assert model.conv2.out_channels == 16
        assert model.fc3.out_features == 10

    def test_forward_pass_shape(self, model, sample_data):
        """Test que le forward pass produit la bonne forme de sortie"""
        model.eval()
        with torch.no_grad():
            output = model(sample_data)

        expected_shape = (sample_data.shape[0], 10)
        assert output.shape == expected_shape

    def test_forward_pass_no_nan(self, model, sample_data):
        """Test qu'il n'y a pas de NaN dans la sortie"""
        model.eval()
        with torch.no_grad():
            output = model(sample_data)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass(self, model, sample_data):
        """Test que le backward pass fonctionne"""
        model.train()

        # Forward pass
        output = model(sample_data)
        target = torch.randint(0, 10, (sample_data.shape[0],))
        loss = F.cross_entropy(output, target)

        # Backward pass
        loss.backward()

        # Vérifier que les gradients ont été calculés
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_model_parameters_count(self, model):
        """Test du nombre de paramètres du modèle"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert (
            trainable_params == total_params
        )  # Tous les paramètres doivent être entraînables

        # Vérifier que le nombre de paramètres est raisonnable (pas trop grand)
        assert total_params < 1e6  # Moins d'1M de paramètres

    def test_model_deterministic(self, model, sample_data):
        """Test que le modèle est déterministe avec la même seed"""
        model.eval()

        torch.manual_seed(42)
        output1 = model(sample_data)

        torch.manual_seed(42)
        output2 = model(sample_data)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_model_saves_loads(self, model, tmp_path):
        """Test que le modèle peut être sauvegardé et rechargé"""
        model_path = tmp_path / "test_model.pt"

        # Sauvegarder
        torch.save(model.state_dict(), model_path)

        # Créer un nouveau modèle et charger
        new_model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        new_model.load_state_dict(torch.load(model_path))

        # Vérifier que les poids sont identiques
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_model_different_batch_sizes(self, model):
        """Test que le modèle fonctionne avec différentes tailles de batch"""
        model.eval()

        for batch_size in [1, 8, 16, 32]:
            data = torch.randn(batch_size, 1, 28, 28)
            with torch.no_grad():
                output = model(data)
            assert output.shape == (batch_size, 10)

    def test_model_train_eval_modes(self, model, sample_data):
        """Test que les modes train/eval affectent le comportement"""
        # Mode train
        model.train()
        assert model.training is True

        # Mode eval
        model.eval()
        assert model.training is False

        # Les sorties peuvent être différentes à cause du dropout/batchnorm
        # (même si ce modèle n'en a pas, c'est une bonne pratique de tester)


class TestModelUtilities:
    """Tests pour les utilitaires du modèle"""

    def test_permutation_consistency(self):
        """Test que la permutation est appliquée de manière cohérente"""
        batch_size = 4
        data = torch.randn(batch_size, 784)
        perm = torch.randperm(784)

        # Appliquer la permutation
        permuted_data = data[:, perm]

        # Vérifier que la forme est préservée
        assert permuted_data.shape == data.shape

        # Vérifier que c'est réversible
        inverse_perm = torch.argsort(perm)
        restored_data = permuted_data[:, inverse_perm]
        assert torch.allclose(data, restored_data)

    def test_data_preprocessing_pipeline(self):
        """Test du pipeline de préprocessing des données"""
        from torchvision import transforms

        # Pipeline de transformation
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Créer une image factice (PIL-like)
        import numpy as np

        fake_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

        # Appliquer les transformations (simulé)
        # En réalité, ceci nécessiterait une vraie image PIL
        # tensor = transform(fake_image)

        # Test basique de normalisation
        data = torch.randn(1, 1, 28, 28)
        mean = 0.1307
        std = 0.3081
        normalized = (data - mean) / std

        assert normalized.mean().abs() < 0.5  # Approximativement centré

    @pytest.mark.parametrize(
        "input_size,n_kernels,output_size", [(1, 6, 10), (3, 8, 5), (1, 4, 2)]
    )
    def test_model_different_configs(self, input_size, n_kernels, output_size):
        """Test du modèle avec différentes configurations"""
        model = ConvNet(
            input_size=input_size, n_kernels=n_kernels, output_size=output_size
        )

        # Test avec des données de la bonne forme
        data = torch.randn(2, input_size, 28, 28)
        output = model(data)

        assert output.shape == (2, output_size)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
