"""
API FastAPI pour le serving du modèle de classification d'images MNIST.
"""
import io
import multiprocessing
import os

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

# Import du modèle
# pylint: disable=E0402
from src.models.convnet import ConvNet

# Désactiver le multiprocessing
multiprocessing.set_start_method("spawn", force=True)

app = FastAPI()

# Initialisation du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle avec ses paramètres
model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "convnet.pt"
)
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)

    # Vérifier le format du modèle sauvegardé
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Nouveau format avec métadonnées
        N_KERNELS = checkpoint.get("n_kernels", 6)
        INPUT_SIZE = checkpoint.get("input_size", 1)
        OUTPUT_SIZE = checkpoint.get("output_size", 10)
        PERMUTATION = checkpoint.get("permutation", torch.randperm(784))

        model = ConvNet(INPUT_SIZE, N_KERNELS, OUTPUT_SIZE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Ancien format (juste les weights)
        print("Chargement d'un ancien modèle sans permutation sauvegardée")
        N_KERNELS = 6
        INPUT_SIZE = 1
        OUTPUT_SIZE = 10
        PERMUTATION = torch.randperm(784)  # Permutation aléatoire

        model = ConvNet(INPUT_SIZE, N_KERNELS, OUTPUT_SIZE)
        model.load_state_dict(checkpoint)
else:
    raise FileNotFoundError(
        f"❌ Aucun modèle entraîné trouvé à {model_path}!\n"
        f"🚀 Entraînez d'abord avec Kedro: cd ../kedro && kedro run\n"
        f"📁 Puis copiez le modèle: cp ../kedro/data/06_models/convnet.pt models/"
    )

model.to(device)
model.eval()


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Prétraite les octets d'une image pour le modèle."""
    # Convertir l'image en PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    # Convertir en niveaux de gris si nécessaire
    if image.mode != "L":
        image = image.convert("L")
    # Redimensionner à 28x28
    image = image.resize((28, 28))

    # Convertir en numpy array et normaliser avec les paramètres MNIST
    image_array = np.array(image, dtype=np.float32) / 255.0
    # Inverser si nécessaire (MNIST a un fond noir)
    if image_array.mean() > 0.5:  # Si l'image est plus claire (fond blanc)
        image_array = 1.0 - image_array

    # Normalisation MNIST
    image_array = (image_array - 0.1307) / 0.3081

    # Convertir en tensor et appliquer la permutation
    image_tensor = torch.from_numpy(image_array.flatten())
    image_permuted = image_tensor[PERMUTATION]
    image_reshaped = image_permuted.view(1, 28, 28)

    return image_reshaped


def predict(image_tensor: torch.Tensor, pred_model: torch.nn.Module) -> dict:
    """Effectue une prédiction sur un tensor d'image."""
    # Ajouter la dimension batch
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Faire la prédiction
    with torch.no_grad():
        output = pred_model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "predicted_class": int(predicted_class),
        "confidence": float(confidence),
        "probabilities": probabilities[0].cpu().numpy().tolist(),
    }


@app.get("/")
async def root():
    """Endpoint racine de l'API."""
    return {
        "message": "API de classification MNIST",
        "endpoints": {
            "predict": "/api/v1/predict",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.post("/api/v1/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Endpoint pour la prédiction d'image."""
    # Lire l'image
    image_bytes = await file.read()

    # Prétraiter l'image (retourne maintenant un tensor avec permutation)
    image_tensor = preprocess_image(image_bytes)

    # Faire la prédiction
    result = predict(image_tensor, model)

    return result


@app.get("/health")
async def health():
    """Endpoint de health check."""
    return {"status": "healthy", "service": "mnist-backend"}


@app.get("/api/info")
async def info():
    """Endpoint d'informations sur l'API."""
    return {"version": "1.0.0", "model": "ConvNet"}
