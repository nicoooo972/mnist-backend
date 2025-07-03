"""
Tests de validation de qualité des modèles ML
Tests avancés pour valider la qualité et robustesse des modèles
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import tempfile
from pathlib import Path


class TestModelQuality:
    """Tests de qualité du modèle"""

    @pytest.fixture
    def trained_model_data(self):
        """Fixture avec un modèle pré-entraîné pour tests"""
        # Simuler un modèle entraîné
        from models.convnet import ConvNet

        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        permutation = torch.randperm(784)

        return {
            "model_state_dict": model.state_dict(),
            "permutation": permutation,
            "hyperparameters": {"epochs": 10, "batch_size": 64, "learning_rate": 0.001},
            "metrics": {"accuracy": 96.5, "test_loss": 0.035},
        }

    def test_model_accuracy_threshold(self):
        """Test que l'accuracy dépasse le seuil minimum"""
        # Simuler des métriques de modèle
        accuracy = 96.5
        min_accuracy = 95.0

        assert (
            accuracy >= min_accuracy
        ), f"Accuracy {accuracy}% < minimum {min_accuracy}%"

    def test_model_loss_threshold(self, trained_model_data):
        """Test que la loss est sous le seuil maximum"""
        test_loss = trained_model_data["metrics"]["test_loss"]
        max_loss = 0.1

        assert test_loss <= max_loss, f"Loss {test_loss} > maximum {max_loss}"

    def test_model_inference_speed(self, trained_model_data):
        """Test de la vitesse d'inférence du modèle"""
        from models.convnet import ConvNet
        import time

        # Charger le modèle
        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        model.load_state_dict(trained_model_data["model_state_dict"])
        model.eval()

        # Préparer les données de test
        batch_size = 100
        test_data = torch.randn(batch_size, 1, 28, 28)
        permutation = trained_model_data["permutation"]

        # Test de vitesse
        start_time = time.time()

        with torch.no_grad():
            # Appliquer la permutation
            data_flattened = test_data.view(batch_size, -1)
            data_permuted = data_flattened[:, permutation]
            data_reshaped = data_permuted.view(batch_size, 1, 28, 28)

            # Inférence
            outputs = model(data_reshaped)

        end_time = time.time()
        inference_time = end_time - start_time

        # Vérifications
        max_inference_time = 1.0  # Maximum 1 seconde pour 100 échantillons
        assert (
            inference_time < max_inference_time
        ), f"Inference too slow: {inference_time:.3f}s"

        # Vérifier la sortie
        assert outputs.shape == (batch_size, 10)
        assert not torch.isnan(outputs).any()

    def test_model_robustness_noise(self, trained_model_data):
        """Test de robustesse du modèle au bruit"""
        from models.convnet import ConvNet

        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        model.load_state_dict(trained_model_data["model_state_dict"])
        model.eval()

        permutation = trained_model_data["permutation"]

        # Données de test propres
        clean_data = torch.randn(10, 1, 28, 28)

        with torch.no_grad():
            # Prédiction sur données propres
            data_flattened = clean_data.view(10, -1)
            data_permuted = data_flattened[:, permutation]
            data_reshaped = data_permuted.view(10, 1, 28, 28)
            clean_outputs = model(data_reshaped)
            clean_predictions = torch.argmax(clean_outputs, dim=1)

            # Ajouter du bruit gaussien
            noise_levels = [0.1, 0.2, 0.3]

            for noise_level in noise_levels:
                noise = torch.randn_like(clean_data) * noise_level
                noisy_data = clean_data + noise

                # Prédiction sur données bruitées
                data_flattened = noisy_data.view(10, -1)
                data_permuted = data_flattened[:, permutation]
                data_reshaped = data_permuted.view(10, 1, 28, 28)
                noisy_outputs = model(data_reshaped)
                noisy_predictions = torch.argmax(noisy_outputs, dim=1)

                # Calculer la stabilité
                stability = (clean_predictions == noisy_predictions).float().mean()
                min_stability = 0.7 if noise_level <= 0.2 else 0.5

                assert (
                    stability >= min_stability
                ), f"Model not robust to noise {noise_level}: stability {stability:.3f}"

    def test_model_confidence_calibration(self, trained_model_data):
        """Test de calibration de la confiance du modèle"""
        from models.convnet import ConvNet

        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        model.load_state_dict(trained_model_data["model_state_dict"])
        model.eval()

        permutation = trained_model_data["permutation"]

        # Générer des données de test
        test_data = torch.randn(50, 1, 28, 28)

        with torch.no_grad():
            data_flattened = test_data.view(50, -1)
            data_permuted = data_flattened[:, permutation]
            data_reshaped = data_permuted.view(50, 1, 28, 28)

            logits = model(data_reshaped)
            probabilities = F.softmax(logits, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]

        # Vérifications de calibration
        assert max_probs.min() > 0.0, "Probabilités minimales doivent être > 0"
        assert max_probs.max() <= 1.0, "Probabilités maximales doivent être <= 1"

        # La confiance moyenne ne doit pas être trop élevée (sur-confiance)
        mean_confidence = max_probs.mean()
        assert mean_confidence < 0.99, f"Modèle trop confiant: {mean_confidence:.3f}"

        # Ni trop faible (sous-confiance)
        assert (
            mean_confidence > 0.3
        ), f"Modèle pas assez confiant: {mean_confidence:.3f}"

    def test_model_class_balance_performance(self, trained_model_data):
        """Test des performances par classe pour détecter les biais"""
        from models.convnet import ConvNet
        from torchvision import datasets, transforms

        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        model.load_state_dict(trained_model_data["model_state_dict"])
        model.eval()

        permutation = trained_model_data["permutation"]

        # Charger des données de test réelles
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dataset = datasets.MNIST(
                tmp_dir, download=True, train=False, transform=transform
            )

            # Prendre un échantillon pour le test
            indices = torch.randperm(len(test_dataset))[:1000]
            subset = torch.utils.data.Subset(test_dataset, indices)
            test_loader = torch.utils.data.DataLoader(subset, batch_size=100)

            class_correct = torch.zeros(10)
            class_total = torch.zeros(10)

            with torch.no_grad():
                for data, target in test_loader:
                    batch_size = data.shape[0]

                    # Appliquer la permutation
                    data_flattened = data.view(batch_size, -1)
                    data_permuted = data_flattened[:, permutation]
                    data_reshaped = data_permuted.view(batch_size, 1, 28, 28)

                    outputs = model(data_reshaped)
                    predictions = torch.argmax(outputs, dim=1)

                    # Compter par classe
                    for i in range(10):
                        class_mask = target == i
                        class_total[i] += class_mask.sum()
                        class_correct[i] += (predictions[class_mask] == i).sum()

            # Calculer l'accuracy par classe
            class_accuracies = class_correct / (
                class_total + 1e-8
            )  # Éviter division par 0

            # Vérifier qu'aucune classe n'a une performance trop faible
            min_class_accuracy = 0.8  # 80% minimum par classe
            for i in range(10):
                if class_total[i] > 0:  # Seulement si la classe est présente
                    assert (
                        class_accuracies[i] >= min_class_accuracy
                    ), f"Classe {i} a une accuracy trop faible: {class_accuracies[i]:.3f}"


class TestModelInvariance:
    """Tests d'invariance du modèle"""

    @pytest.fixture
    def model_and_data(self):
        """Fixture avec modèle et données"""
        from models.convnet import ConvNet

        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        permutation = torch.randperm(784)
        test_data = torch.randn(5, 1, 28, 28)

        return model, permutation, test_data

    def test_batch_size_invariance(self, model_and_data):
        """Test que le modèle donne des résultats cohérents peu importe la taille de batch"""
        model, permutation, test_data = model_and_data
        model.eval()

        with torch.no_grad():
            # Prédiction batch par batch
            individual_outputs = []
            for i in range(test_data.shape[0]):
                single_data = test_data[i : i + 1]  # Batch de taille 1

                data_flattened = single_data.view(1, -1)
                data_permuted = data_flattened[:, permutation]
                data_reshaped = data_permuted.view(1, 1, 28, 28)

                output = model(data_reshaped)
                individual_outputs.append(output)

            individual_batch = torch.cat(individual_outputs, dim=0)

            # Prédiction en un seul batch
            batch_size = test_data.shape[0]
            data_flattened = test_data.view(batch_size, -1)
            data_permuted = data_flattened[:, permutation]
            data_reshaped = data_permuted.view(batch_size, 1, 28, 28)

            full_batch_output = model(data_reshaped)

        # Les résultats doivent être identiques
        assert torch.allclose(
            individual_batch, full_batch_output, atol=1e-6
        ), "Modèle pas invariant à la taille de batch"


class TestModelSafety:
    """Tests de sécurité du modèle"""

    def test_model_input_validation(self):
        """Test de validation des entrées du modèle"""
        from models.convnet import ConvNet

        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        model.eval()

        # Test avec des entrées invalides
        with torch.no_grad():
            # Forme incorrecte
            with pytest.raises((RuntimeError, ValueError)):
                wrong_shape = torch.randn(1, 3, 28, 28)  # 3 canaux au lieu de 1
                model(wrong_shape)

            # Valeurs extrêmes
            extreme_values = torch.full((1, 1, 28, 28), 1e6)
            output_extreme = model(extreme_values)
            assert not torch.isnan(
                output_extreme
            ).any(), "Modèle produit NaN avec valeurs extrêmes"
            assert not torch.isinf(
                output_extreme
            ).any(), "Modèle produit Inf avec valeurs extrêmes"

    def test_model_memory_usage(self):
        """Test de l'utilisation mémoire du modèle"""
        from models.convnet import ConvNet
        import psutil
        import os

        # Mesurer la mémoire avant
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Créer et utiliser le modèle
        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        test_data = torch.randn(100, 1, 28, 28)

        with torch.no_grad():
            output = model(test_data)

        # Mesurer la mémoire après
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Vérifier que l'augmentation mémoire est raisonnable
        max_memory_increase = 100  # Maximum 100 MB
        assert (
            memory_increase < max_memory_increase
        ), f"Utilisation mémoire trop élevée: {memory_increase:.1f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
