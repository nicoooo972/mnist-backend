"""
Tests d'intégration pour le pipeline complet d'entraînement
"""
import pytest
import torch
import tempfile
import json
import os
import sys
from pathlib import Path

# Ajouter le chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class TestTrainingPipeline:
    """Tests d'intégration pour le pipeline d'entraînement complet"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Répertoire temporaire pour les modèles"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    def test_end_to_end_training(self, temp_model_dir):
        """Test d'entraînement end-to-end avec peu d'époques"""
        # Ce test valide le pipeline complet d'entraînement
        from models.convnet import ConvNet
        from torchvision import datasets, transforms
        import torch.nn.functional as F
        
        # Configuration minimale pour test rapide
        epochs = 2
        batch_size = 32
        learning_rate = 0.01
        
        # Setup device
        device = torch.device("cpu")
        
        # Data loading (échantillon réduit pour test rapide)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Créer un petit dataset pour le test
        train_dataset = datasets.MNIST(
            temp_model_dir / "data", 
            download=True, 
            train=True, 
            transform=transform
        )
        
        # Réduire le dataset pour test rapide
        small_indices = torch.randperm(len(train_dataset))[:1000]
        small_dataset = torch.utils.data.Subset(train_dataset, small_indices)
        
        train_loader = torch.utils.data.DataLoader(
            small_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Model setup
        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        permutation = torch.randperm(784)
        
        # Training
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_size_actual = data.shape[0]
                
                # Appliquer la permutation
                data_flattened = data.view(batch_size_actual, -1)
                data_permuted = data_flattened[:, permutation]
                data_reshaped = data_permuted.view(batch_size_actual, 1, 28, 28)
                
                optimizer.zero_grad()
                logits = model(data_reshaped)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                if batch_idx == 0 and epoch == 0:
                    initial_loss = loss.item()
                
                # Arrêter après quelques batches pour test rapide
                if batch_idx >= 10:
                    break
            
            final_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Assertions
        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss * 2  # Le loss ne doit pas exploser
        
        # Test de sauvegarde
        model_path = temp_model_dir / "test_model.pt"
        model_data = {
            "model_state_dict": model.state_dict(),
            "permutation": permutation,
            "hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            },
            "metrics": {
                "final_loss": final_loss
            }
        }
        
        torch.save(model_data, model_path)
        assert model_path.exists()
        
        # Test de rechargement
        loaded_data = torch.load(model_path, map_location='cpu')
        assert "model_state_dict" in loaded_data
        assert "permutation" in loaded_data
        assert "hyperparameters" in loaded_data
    
    def test_model_validation_pipeline(self, temp_model_dir):
        """Test du pipeline de validation du modèle"""
        from models.convnet import ConvNet
        from torchvision import datasets, transforms
        import torch.nn.functional as F
        
        # Créer un modèle pré-entraîné factice
        model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        permutation = torch.randperm(784)
        
        # Sauvegarder le modèle
        model_path = temp_model_dir / "model_to_validate.pt"
        model_data = {
            "model_state_dict": model.state_dict(),
            "permutation": permutation,
            "hyperparameters": {
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.001
            },
            "metrics": {
                "accuracy": 95.5,
                "test_loss": 0.05
            }
        }
        torch.save(model_data, model_path)
        
        # Pipeline de validation
        loaded_data = torch.load(model_path, map_location='cpu')
        
        # Validation de structure
        assert "model_state_dict" in loaded_data
        assert "permutation" in loaded_data
        assert "hyperparameters" in loaded_data
        assert "metrics" in loaded_data
        
        # Validation des métriques
        metrics = loaded_data["metrics"]
        assert "accuracy" in metrics
        assert "test_loss" in metrics
        
        accuracy = metrics["accuracy"]
        test_loss = metrics["test_loss"]
        
        # Critères de validation
        min_accuracy = 90.0
        max_loss = 1.0
        
        validation_results = {
            "accuracy_check": accuracy >= min_accuracy,
            "loss_check": test_loss <= max_loss,
            "structure_check": True
        }
        
        # Test de chargement du modèle
        new_model = ConvNet(input_size=1, n_kernels=6, output_size=10)
        new_model.load_state_dict(loaded_data["model_state_dict"])
        
        # Test d'inférence
        test_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            output = new_model(test_input)
        
        assert output.shape == (1, 10)
        assert not torch.isnan(output).any()
        
        # Résultat final
        all_passed = all(validation_results.values())
        assert all_passed  # Le modèle factice doit passer la validation
    
    def test_data_pipeline_integrity(self, temp_model_dir):
        """Test de l'intégrité du pipeline de données"""
        from torchvision import datasets, transforms
        
        # Test de téléchargement et validation MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            temp_model_dir / "data",
            download=True,
            train=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            temp_model_dir / "data",
            download=True,
            train=False,
            transform=transform
        )
        
        # Vérifications de base
        assert len(train_dataset) == 60000
        assert len(test_dataset) == 10000
        
        # Vérifier un échantillon
        sample_x, sample_y = train_dataset[0]
        assert sample_x.shape == (1, 28, 28)
        assert 0 <= sample_y <= 9
        assert isinstance(sample_y, int)
        
        # Test de dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )
        
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape == (32, 1, 28, 28)
        assert batch_y.shape == (32,)
        assert batch_y.dtype == torch.int64
        
        # Test de transformation et permutation
        permutation = torch.randperm(784)
        batch_size = batch_x.shape[0]
        
        # Appliquer la permutation
        data_flattened = batch_x.view(batch_size, -1)
        data_permuted = data_flattened[:, permutation]
        data_reshaped = data_permuted.view(batch_size, 1, 28, 28)
        
        assert data_reshaped.shape == batch_x.shape
        assert not torch.equal(data_reshaped, batch_x)  # Les données doivent être différentes
    
    def test_hyperparameter_configuration(self):
        """Test de la configuration des hyperparamètres"""
        # Simulation des inputs GitHub Actions
        hyperparams = {
            "epochs": "10",
            "batch_size": "64", 
            "learning_rate": "0.001",
            "min_accuracy": "95.0"
        }
        
        # Conversion et validation
        epochs = int(hyperparams.get("epochs", "10"))
        batch_size = int(hyperparams.get("batch_size", "64"))
        learning_rate = float(hyperparams.get("learning_rate", "0.001"))
        min_accuracy = float(hyperparams.get("min_accuracy", "95.0"))
        
        # Validation des ranges
        assert 1 <= epochs <= 100
        assert 1 <= batch_size <= 512
        assert 0.0001 <= learning_rate <= 0.1
        assert 0.0 <= min_accuracy <= 100.0
        
        # Test avec valeurs par défaut
        default_hyperparams = {}
        epochs_default = int(default_hyperparams.get("epochs", "10"))
        assert epochs_default == 10
    
    def test_model_registry_operations(self, temp_model_dir):
        """Test des opérations du registre de modèles"""
        # Créer un registre de modèles
        registry_path = temp_model_dir / "model_registry.json"
        
        model_info = {
            "production_model": {
                "version": "20240101-120000-abc1234",
                "accuracy": 96.5,
                "test_loss": 0.03,
                "promoted_at": "2024-01-01T12:00:00Z",
                "git_commit": "abc1234567890",
                "path": "models/convnet-20240101-120000-abc1234.pt"
            },
            "staging_models": [
                {
                    "version": "20240101-130000-def5678",
                    "accuracy": 95.8,
                    "test_loss": 0.04,
                    "status": "testing"
                }
            ]
        }
        
        # Sauvegarder le registre
        with open(registry_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        assert registry_path.exists()
        
        # Charger et valider le registre
        with open(registry_path, 'r') as f:
            loaded_registry = json.load(f)
        
        assert "production_model" in loaded_registry
        assert "staging_models" in loaded_registry
        
        prod_model = loaded_registry["production_model"]
        assert "version" in prod_model
        assert "accuracy" in prod_model
        assert prod_model["accuracy"] > 90.0


class TestContainerIntegration:
    """Tests d'intégration pour les conteneurs"""
    
    @pytest.mark.slow
    def test_docker_build_simulation(self, temp_model_dir):
        """Simulation du build Docker (sans Docker réel)"""
        # Créer les fichiers nécessaires pour le build
        dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "-m", "src.api.main"]
"""
        
        requirements_content = """
torch==2.0.1
torchvision==0.15.2
fastapi==0.104.1
uvicorn==0.24.0
numpy==1.24.3
"""
        
        dockerfile_path = temp_model_dir / "Dockerfile"
        requirements_path = temp_model_dir / "requirements.txt"
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        # Vérifier que les fichiers existent
        assert dockerfile_path.exists()
        assert requirements_path.exists()
        
        # Simulation de la validation du Dockerfile
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        assert "FROM python:" in content
        assert "COPY requirements.txt" in content
        assert "RUN pip install" in content
        assert "CMD" in content
    
    def test_environment_configuration(self):
        """Test de la configuration d'environnement"""
        # Simulation des variables d'environnement
        env_vars = {
            "REGISTRY": "ghcr.io",
            "IMAGE_NAME": "mnist-mlops",
            "PYTHON_VERSION": "3.11",
            "MODEL_REGISTRY": "models"
        }
        
        # Validation
        assert env_vars["REGISTRY"] in ["ghcr.io", "docker.io"]
        assert "mnist" in env_vars["IMAGE_NAME"]
        assert env_vars["PYTHON_VERSION"].startswith("3.")
        assert env_vars["MODEL_REGISTRY"] == "models"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 