# MNIST Backend - API de Serving

Ce projet contient une API FastAPI conçue pour servir un modèle de reconnaissance de chiffres manuscrits (MNIST).

## Rôle dans l'architecture MLOps

Ce service constitue la brique de **Serving** de notre architecture. Il ne gère pas l'entraînement du modèle. Sa seule responsabilité est de :
1.  Charger un artefact de modèle pré-entraîné (`convnet.pt`).
2.  Exposer un endpoint `/predict` qui accepte une image de chiffre et retourne une prédiction.

## Automatisation (CI/CD)

Le workflow GitHub Actions configuré dans ce dépôt est un exemple de **Continuous Delivery**. À chaque push sur la branche `main` :
1.  **Récupération du Modèle** : Le workflow va automatiquement chercher le dernier modèle de production entraîné et publié par le projet `kedro-backend`.
2.  **Build de l'Image** : Il intègre ce modèle dans une nouvelle image Docker de l'API.
3.  **Publication de l'Image** : Il publie cette image sur le GitHub Container Registry (`ghcr.io`).

Cette image est ensuite prête à être déployée par le projet `mnist-deployment`.

## 🚀 Démarrage rapide

```bash
# Installation des dépendances
pip install -r requirements.txt

# Entraînement du modèle (optionnel si le modèle existe déjà)
python -m src.train_model

# Lancement de l'API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Endpoints

- `POST /api/v1/predict` - Classification d'image
- `GET /health` - Statut de l'API  
- `GET /api/info` - Informations sur l'API
- `GET /docs` - Documentation Swagger

## 🐳 Docker

```bash
docker build -t mnist-backend .
docker run -p 8000:8000 mnist-backend
```

## 📁 Structure

```
src/
├── api/
│   └── main.py          # API FastAPI
├── models/
│   ├── __init__.py
│   └── convnet.py       # Modèle ConvNet
└── train_model.py       # Script d'entraînement
models/
└── convnet.pt          # Modèle entraîné
``` 